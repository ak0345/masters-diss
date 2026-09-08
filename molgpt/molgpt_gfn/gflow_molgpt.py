"""Training loop for a guide over the frozen MolGPT prior.

Adapted from gflow.py's `LitGFlowNet`, scoped to plan_molgpt.md's pilot grid
(objective in {db, rtb}, replay off, no tempgain, no revkl/fwdkl). Ported
faithfully where the interface is architecture-agnostic (the DB loss math,
the RTB loss, the guide dispatch pattern, the `roll` dict shape); genuinely
new where Quetzal's coordinate machinery has no MolGPT equivalent (the
rollout builders below are plain autoregressive-token loops, no coordinate
diffusion -- see plan_molgpt.md, "Rollout builder").

Deliberate simplifications relative to gflow.py, noted so nobody mistakes
them for oversights:

  - **No PyTorch Lightning.** gflow.py is a LightningModule for
    multi-GPU/EMA/checkpoint-callback convenience Quetzal's larger runs need.
    A MolGPT guide run is small (single GPU, plan_molgpt.md estimates under
    30 minutes) -- a plain training loop avoids a large dependency and a
    Lightning-config surface nothing here exercises. Checkpoints are a plain
    `torch.save` dict (see `save_checkpoint`/`load_checkpoint` below), not a
    Lightning `.ckpt`; `flip_molgpt.py` reads this format, not
    `final_dump.py`'s.
  - **Guide EMA, added 2026-09-04.** Originally skipped ("small run, keep it
    simple"), then added back once it turned out to matter for comparability:
    the Quetzal numbers this pilot is measured against
    (`results/flips-guide/_aggs/`) are all `guide_source=ema`, and
    `final_dump.py`'s default eval convention (N=5000, `guide_source=ema`,
    `sample_temp=cfg.sample_temp`, `rand_eps=0`) evaluates the EMA guide, not
    the raw trained one -- ported verbatim from `gflow.py`:
    `torch.optim.swa_utils.AveragedModel(guide, device="cpu",
    multi_avg_fn=get_ema_multi_avg_fn(cfg.guide_ema_decay), use_buffers=True)`,
    updated via `.update_parameters(guide)` after every training step. See
    `papers/current/decisions.md`, 2026-09-04, and this leg's
    `../results/final_dump.py`-equivalent, `final_dump_molgpt.py`.
  - **No hang-guard.** gflow.py's reward calls can hang (RDKit bond
    perception from 3D geometry, an untimed xtb subprocess) which is what
    `hang_guard.py` exists to catch. MolGPT's reward path
    (`reward_adapter.py`) is `Chem.MolFromSmiles` plus a GuacaMol scorer --
    no geometry step, much less hang-prone -- but this is an assumption, not
    a proof; if a sweep run does hang, that assumption was wrong and this
    needs a guard adding, not silently trusting it.
  - **Replay is refused, not silently ignored** (`cfg.use_replay=True`
    raises `NotImplementedError`) -- see plan_molgpt.md, "Replay buffer": out
    of scope for this pilot, and `replay_buffer.py`'s storage format is
    Quetzal-shaped (atoms+coords), not a token-only shape this could reuse
    directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GFNConfig  # noqa: E402
from molgpt_prior import FrozenMolGPT, MolGPTConfig  # noqa: E402
from guides import HiddenGuide, LogitGuide, LogFlowHead, masked_guided_logits_dispatch  # noqa: E402
from reward_adapter import build_reward_smiles  # noqa: E402


def _decode(prior: FrozenMolGPT, token_row) -> str:
    """Token ids (post-seed-token) -> SMILES string, truncated at the first
    PAD/EOS token `<` -- the boundary `stop_step` marks, unlike
    devalab/molgpt's own `generate.py` which strips `<` wherever it occurs
    (fine for a completed, fully-decoded rollout, but truncation is the
    operationally correct read here since we also need `stop_step` to line
    up with where the string actually ends)."""
    out = []
    for i in token_row:
        if i == prior.pad_idx:
            break
        out.append(prior.itos[i])
    return "".join(out)


class Trainer:
    def __init__(self, cfg: GFNConfig, device: str = "cuda", ckpt_root: str = "logs/molgpt-gfn"):
        self.cfg = cfg
        self.device = device
        # Matches plan_molgpt.md's directory layout ("logs/molgpt-gfn/ run
        # directories, same naming convention as logs/quetzal-gfn") and what
        # scripts_molgpt/01_train_guides.sh's skip-check globs against.
        self.ckpt_root = Path(ckpt_root)

        molcfg = MolGPTConfig(vocab_size=cfg.vocab_size, block_size=cfg.max_len)
        self.prior = FrozenMolGPT(molcfg, cfg.molgpt_ckpt, device=device)
        d_model = molcfg.n_embd

        torch.manual_seed(cfg.seed)
        if cfg.use_hidden_guide:
            self.guide = HiddenGuide(
                d_model, self.prior.proj_logits, hidden=cfg.guide_hidden,
                layers=cfg.guide_layers, vocab_size=cfg.vocab_size,
                also_output_residual=cfg.hidden_guide_out_residual,
            ).to(device)
        else:
            self.guide = LogitGuide(d_model, cfg.vocab_size, cfg.guide_hidden, cfg.guide_layers).to(device)

        # Sanity check mirroring gflow.py's own assert: the guide must be the
        # identity at initialisation (both branches zero-init their last
        # layer). Probed on a REAL hidden state from the prior, not a zero
        # tensor -- a zero probe would pass trivially even with broken wiring
        # (e.g. proj_logits ignoring its input).
        with torch.no_grad():
            seed = torch.full((2, 1), self.prior.seed_idx, dtype=torch.long, device=device)
            h_probe = self.prior.hidden_states(seed)[:, -1, :]
            if isinstance(self.guide, HiddenGuide):
                g0 = self.guide.guided_logits(h_probe)
            else:
                g0 = self.prior.proj_logits(h_probe) + self.guide(h_probe)
            p0 = self.prior.proj_logits(h_probe)
            if not torch.allclose(g0, p0, atol=1e-4):
                raise AssertionError(
                    "guide is not the identity at init -- guided logits differ "
                    f"from prior logits by up to {(g0 - p0).abs().max().item():.4g}. "
                    "This should be impossible given zero-init last layers; "
                    "something is wired wrong before any training has happened.")

        self.flow_head = None
        self.logZ = None
        params = list(self.guide.parameters())
        if cfg.objective == "db":
            self.flow_head = LogFlowHead(d_model, cfg.flow_hidden, cfg.flow_layers).to(device)
            params += list(self.flow_head.parameters())
        elif cfg.objective == "rtb":
            self.logZ = torch.nn.Parameter(torch.tensor(float(cfg.logz_init), device=device))
        else:
            raise ValueError(
                f"gflow_molgpt.py only implements objective in ('db','rtb') "
                f"per plan_molgpt.md's pilot scope, got {cfg.objective!r}.")

        if cfg.use_replay:
            raise NotImplementedError(
                "Replay is out of scope for this pilot (plan_molgpt.md: 'Replay: "
                "off only') and there is no MolGPT-shaped replay buffer "
                "implementation yet. Leave --use_replay unset.")
        if cfg.warm_start_guide:
            raise NotImplementedError(
                "warm_start_guide is not wired up here -- gflow.py's "
                "_load_warm_start has no MolGPT port yet. Leave it unset "
                "rather than silently training from scratch while the flag "
                "looks honoured.")

        opt_groups = [{"params": params, "lr": cfg.lr}]
        if self.logZ is not None:
            opt_groups.append({"params": [self.logZ], "lr": cfg.logz_lr})
        self.opt = torch.optim.AdamW(opt_groups, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.wd)

        # Ported verbatim from gflow.py -- see this file's module docstring,
        # "Guide EMA, added 2026-09-04".
        self.guide_ema = AveragedModel(
            self.guide, device="cpu",
            multi_avg_fn=get_ema_multi_avg_fn(cfg.guide_ema_decay),
            use_buffers=True,
        )
        self.guide_ema.eval()
        for p in self.guide_ema.parameters():
            p.requires_grad = False

        self.reward_fn = build_reward_smiles(cfg)
        self.step = 0

    # ---------------------------------------------------------------- rollouts

    def _prior_step(self, tokens):
        """tokens: [B, t] -> (h: [B,d_model], prior_logits: [B,vocab]), no_grad
        (the prior is frozen; this is a compute optimisation, not required for
        correctness since frozen params already carry requires_grad=False)."""
        with torch.no_grad():
            h = self.prior.hidden_states(tokens)[:, -1, :]
            prior_logits = self.prior.proj_logits(h)
        return h, prior_logits

    def generate_guided(self, bsz, guide, sample_temp=1.0, rand_eps=0.0, max_len=None):
        """On-policy rollout under `guide`. Gradient-attached through the
        guide (NOT wrapped in a blanket `torch.no_grad()`): only the frozen
        prior's forward pass is no_grad (see `_prior_step`); the guide's own
        `log_softmax` output stays attached, which is what RTB needs to
        backprop through `logp_policy`. Callers that want a detached eval
        rollout should wrap the call site in `torch.no_grad()` themselves --
        an ambient no_grad context silences the inner grad regardless.

        Mirrors gflow.py's `_generate_guided`, minus the coords/encode2/
        sample_coord block (no geometry here)."""
        max_len = max_len or self.cfg.max_len
        device = self.device
        x = torch.full((bsz, 1), self.prior.seed_idx, dtype=torch.long, device=device)
        alive = torch.ones(bsz, dtype=torch.bool, device=device)
        logp_policy = torch.zeros(bsz, device=device)
        logp_prior = torch.zeros(bsz, device=device)

        for _t in range(max_len - 1):
            h, prior_logits = self._prior_step(x)
            guided = masked_guided_logits_dispatch(guide, prior_logits, h, self.prior.atom_token_mask)
            logp_pol = F.log_softmax(guided, dim=-1)
            logp_pri = F.log_softmax(prior_logits, dim=-1)
            behav = F.softmax(guided.detach() / sample_temp, dim=-1)
            if rand_eps > 0:
                behav = (1 - rand_eps) * behav + rand_eps / behav.shape[-1]
            nxt = torch.multinomial(behav, 1).squeeze(-1)

            logp_policy = logp_policy + logp_pol.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) * alive
            logp_prior = logp_prior + logp_pri.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) * alive
            alive = alive & (nxt != self.prior.pad_idx)
            x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
            if not alive.any():
                break

        smiles = [_decode(self.prior, row) for row in x[:, 1:].tolist()]
        log_reward = self.compute_log_reward(smiles)
        return {"tokens": x, "smiles": smiles, "logp_policy": logp_policy,
                "logp_prior": logp_prior, "log_reward": log_reward}

    def db_rollout_states(self, guide, bsz, max_len=None):
        """Gradient-attached rollout for the DB objective. Reproduces
        gflow.py's `_db_rollout_states` dict contract exactly: `hs`,
        `logpf_chosen`, `logpf_stop`, `logprior_step`, `step_mask`,
        `stop_step`, all shaped `[B,T]` (`stop_step` is `[B]`), so
        `db_loss_from_rollout` below is a verbatim drop-in.

        `logpf_stop` is `log q(PAD | state)`, evaluated at EVERY step
        regardless of what was actually sampled -- the DB terminal residual
        needs "probability the guide would have stopped here," evaluated at
        the state where a stop actually occurred, matching gflow.py's
        `logp_pol[:, STOP]` convention with STOP replaced by MolGPT's PAD
        index (MolGPT has no separate STOP/PAD distinction -- see
        `../vendor/PROVENANCE.md`)."""
        max_len = max_len or self.cfg.max_len
        device = self.device
        x = torch.full((bsz, 1), self.prior.seed_idx, dtype=torch.long, device=device)
        alive = torch.ones(bsz, dtype=torch.bool, device=device)
        stop_step = torch.full((bsz,), -1, dtype=torch.long, device=device)

        hs_list, logpf_chosen_list, logpf_stop_list = [], [], []
        logprior_step_list, step_mask_list = [], []

        T = max_len - 1
        for t in range(T):
            h, prior_logits = self._prior_step(x)
            guided = masked_guided_logits_dispatch(guide, prior_logits, h, self.prior.atom_token_mask)   # gradient-attached
            logp_pol = F.log_softmax(guided, dim=-1)
            with torch.no_grad():
                logp_pri = F.log_softmax(prior_logits, dim=-1)
                behav = F.softmax(guided.detach() / self.cfg.sample_temp, dim=-1)
                if self.cfg.rand_eps > 0:
                    behav = (1 - self.cfg.rand_eps) * behav + self.cfg.rand_eps / behav.shape[-1]
                nxt = torch.multinomial(behav, 1).squeeze(-1)

            hs_list.append(h)
            logpf_chosen_list.append(logp_pol.gather(-1, nxt.unsqueeze(-1)).squeeze(-1))
            logpf_stop_list.append(logp_pol[:, self.prior.pad_idx])
            logprior_step_list.append(logp_pri.gather(-1, nxt.unsqueeze(-1)).squeeze(-1).detach())
            step_mask_list.append(alive.clone())

            just_stopped = alive & (nxt == self.prior.pad_idx)
            stop_step = torch.where(just_stopped, torch.full_like(stop_step, t), stop_step)
            alive = alive & (nxt != self.prior.pad_idx)
            x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
            if not alive.any():
                # Pad the remaining steps so every tensor stays a rectangular
                # [B,T]: step_mask=False there means db_loss_from_rollout's
                # interior_valid excludes them, so the padding value itself
                # (a repeat of the last real step) is inert, never scored.
                for _ in range(T - len(hs_list)):
                    hs_list.append(h)
                    logpf_chosen_list.append(logpf_chosen_list[-1])
                    logpf_stop_list.append(logpf_stop_list[-1])
                    logprior_step_list.append(logprior_step_list[-1])
                    step_mask_list.append(torch.zeros_like(alive))
                break

        stop_step = torch.where(stop_step < 0, torch.full_like(stop_step, T - 1), stop_step)
        smiles = [_decode(self.prior, row) for row in x[:, 1:].tolist()]
        return {
            "hs": torch.stack(hs_list, dim=1),
            "logpf_chosen": torch.stack(logpf_chosen_list, dim=1),
            "logpf_stop": torch.stack(logpf_stop_list, dim=1),
            "logprior_step": torch.stack(logprior_step_list, dim=1),
            "step_mask": torch.stack(step_mask_list, dim=1),
            "stop_step": stop_step,
            "smiles": smiles,
            "tokens": x,
        }

    @staticmethod
    def db_loss_from_rollout(flow_head, roll, log_reward, beta, invalid_logr=None, target_clip=None):
        """Ported verbatim from gflow.py's `LitGFlowNet.db_loss_from_rollout`
        -- it never touches `self` beyond its explicit arguments, so this is
        a faithful drop-in against the `roll` dict `db_rollout_states` builds
        above, not a reimplementation. See gflow.py:897 for the original."""
        hs = roll["hs"]
        logpf_chosen = roll["logpf_chosen"]
        logpf_stop = roll["logpf_stop"]
        logprior_step = roll["logprior_step"]
        step_mask = roll["step_mask"]
        stop_step = roll["stop_step"]
        B, T, _d = hs.shape
        device = hs.device

        f = flow_head(hs)
        logprior_partial = torch.cumsum(logprior_step, dim=1) - logprior_step
        logF = logprior_partial + f
        logF_next = torch.cat([logF[:, 1:], logF[:, -1:].detach()], dim=1)

        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        interior_valid = (t_idx < stop_step.unsqueeze(1)) & step_mask
        interior_res = (logF + logpf_chosen - logF_next) * interior_valid.float()
        n_int = interior_valid.float().sum().clamp(min=1.0)
        interior_loss = interior_res.pow(2).sum() / n_int

        x = stop_step.clamp(min=0, max=T - 1)
        bidx = torch.arange(B, device=device)
        logF_x = logF[bidx, x]
        logpf_stop_x = logpf_stop[bidx, x]
        logprior_partial_x = logprior_partial[bidx, x]

        logR = log_reward
        if target_clip is not None:
            logR = logR.clamp(min=target_clip)
        logR_b = beta * logR

        if invalid_logr is None:
            valid_term = torch.ones_like(log_reward, dtype=torch.bool)
        else:
            valid_term = log_reward > (invalid_logr + 0.1)

        terminal_res_full = logF_x + logpf_stop_x - (logprior_partial_x + logR_b)
        if valid_term.any():
            terminal_res = terminal_res_full[valid_term]
            terminal_loss = terminal_res.pow(2).mean()
        else:
            terminal_loss = (terminal_res_full * 0.0).sum()
            terminal_res = terminal_res_full

        total = interior_loss + terminal_loss
        stats = {
            "db/interior_loss": interior_loss.detach().item(),
            "db/terminal_loss": terminal_loss.detach().item() if torch.is_tensor(terminal_loss) else float(terminal_loss),
            "db/terminal_res_abs_mean": terminal_res.detach().abs().mean().item(),
            "db/mean_logF_x": logF_x.detach().mean().item(),
            "db/frac_valid_terminal": valid_term.float().mean().item(),
        }
        return total, stats

    def compute_log_reward(self, smiles_list):
        vals = [self.reward_fn(smi) for smi in smiles_list]
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    # ---------------------------------------------------------------- training

    def training_step(self):
        cfg = self.cfg
        if cfg.objective == "db":
            roll = self.db_rollout_states(self.guide, cfg.bsz)
            log_reward = self.compute_log_reward(roll["smiles"])
            loss, stats = self.db_loss_from_rollout(
                self.flow_head, roll, log_reward, cfg.reward_beta,
                invalid_logr=cfg.invalid_logr, target_clip=cfg.db_target_clip)
        else:  # rtb -- ported verbatim from gflow.py's RTB branch, gflow.py:1070-1082
            out = self.generate_guided(cfg.bsz, self.guide, sample_temp=1.0, rand_eps=0.0)
            ratio = out["logp_policy"] - out["logp_prior"]
            if cfg.ratio_clip > 0:
                ratio = ratio.clamp(-cfg.ratio_clip, cfg.ratio_clip)
            logR = cfg.reward_beta * out["log_reward"]
            delta = ratio - logR
            loss = (self.logZ + delta).pow(2).mean()
            log_reward = out["log_reward"]
            stats = {"rtb/logZ": self.logZ.detach().item(), "rtb/log_ratio_mean": ratio.mean().item()}

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        params = list(self.guide.parameters()) + (list(self.flow_head.parameters()) if self.flow_head else [])
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        self.opt.step()
        self.guide_ema.update_parameters(self.guide)   # every step, both objectives -- matches gflow.py

        valid_frac = float(sum(1 for lr in log_reward.tolist() if lr > cfg.invalid_logr + 0.1) / len(log_reward))
        stats.update({
            "train/loss": loss.detach().item(),
            "train/log_reward_mean": log_reward.mean().item(),
            "train/log_reward_max": log_reward.max().item(),
            "train/reward_valid_frac": valid_frac,
        })
        self.step += 1
        return stats

    def fit(self, log_path: Path = None):
        cfg = self.cfg
        log_f = open(log_path, "a") if log_path else None
        wandb_run = None
        if cfg.wandb:
            import os
            import wandb
            wandb_run = wandb.init(
                project="molgpt-gfn", entity=os.getenv("WANDB_ENTITY"),
                name=cfg.name, config=asdict(cfg))
        t0 = time.time()
        for epoch in range(cfg.max_epochs):
            for _ in range(cfg.steps_per_epoch):
                stats = self.training_step()
                if log_f:
                    log_f.write(json.dumps({"step": self.step, "epoch": epoch, **stats}) + "\n")
                    log_f.flush()
                if wandb_run is not None:
                    wandb_run.log(stats, step=self.step)
            print(f"[epoch {epoch}] step={self.step} "
                  f"loss={stats['train/loss']:.4f} "
                  f"log_reward_mean={stats['train/log_reward_mean']:.4f} "
                  f"valid_frac={stats['train/reward_valid_frac']:.3f} "
                  f"({time.time() - t0:.0f}s elapsed)")
            self.save_checkpoint(epoch=epoch)
        if log_f:
            log_f.close()
        if wandb_run is not None:
            wandb_run.finish()

    # ---------------------------------------------------------------- checkpoints

    def save_checkpoint(self, path: Path = None, epoch: int = None):
        ckpt_dir = self.ckpt_root / self.cfg.name / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = path or (ckpt_dir / "last.ckpt")
        payload = {
            "config": asdict(self.cfg),
            "guide_state_dict": self.guide.state_dict(),
            "guide_ema_state_dict": self.guide_ema.module.state_dict(),
            "guide_class": type(self.guide).__name__,
            "step": self.step,
            "epoch": epoch,
        }
        if self.flow_head is not None:
            payload["flow_head_state_dict"] = self.flow_head.state_dict()
        if self.logZ is not None:
            payload["logZ"] = self.logZ.detach().cpu()
        torch.save(payload, path)


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    for f in fields(GFNConfig):
        flag = f"--{f.name}"
        # `config.py` uses `from __future__ import annotations`, so `f.type`
        # is a string ("bool", "int", ...), not a type object -- dispatch on
        # the default's runtime type instead. isinstance(default, bool) must
        # come before int, since bool is an int subclass in Python.
        if isinstance(f.default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(flag, type=type(f.default) if f.default is not None else str, default=f.default)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt_root", default="logs/molgpt-gfn",
                    help="run-directory root; matches scripts_molgpt/01_train_guides.sh's CKPT_ROOT")
    return p


def parse_args(argv=None):
    args = _build_argparser().parse_args(argv)
    kwargs = {f.name: getattr(args, f.name) for f in fields(GFNConfig)}
    cfg = GFNConfig(**kwargs)
    return cfg, args.device, args.ckpt_root


if __name__ == "__main__":
    cfg, device, ckpt_root = parse_args()
    trainer = Trainer(cfg, device=device, ckpt_root=ckpt_root)
    run_dir = trainer.ckpt_root / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer.fit(log_path=run_dir / "train.jsonl")
