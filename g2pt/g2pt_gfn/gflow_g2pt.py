"""Training loop for a guide over the frozen G2PT prior.

Adapted from `molgpt/molgpt_gfn/gflow_molgpt.py`, which itself adapted
`gflow.py`'s `LitGFlowNet` -- see that file's module docstring for the
deliberate simplifications shared by both legs (no Lightning, no guide EMA,
no hang-guard, replay/warm-start refuse loudly). The one genuinely new piece
here, not present in the MolGPT leg at all, is **atom-step gating**: this
pilot's G2PT guide is Option A (node-type-only, matching Quetzal/MolGPT),
which for G2PT specifically means the guide is applied ONLY when the current
decoding step is a legal atom-type decision (`FrozenG2PT.is_atom_step`) and
left as the untouched frozen-prior distribution everywhere else (`IDX_*`,
`BOND_*`, structural tokens). See `papers/current/decisions.md`, "G2PT leg
uses Option A," for why this is architecturally necessary here (G2PT has no
separate atom-type-only output head the way Quetzal does -- the gating has
to happen in the rollout, not in the model).
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
from g2pt_prior import FrozenG2PT, G2PTConfig, seq_to_smiles  # noqa: E402
from guides import HiddenGuide, LogitGuide, LogFlowHead, masked_guided_logits_dispatch  # noqa: E402
from reward_adapter import build_reward_smiles  # noqa: E402


def _decode(prior: FrozenG2PT, token_row) -> str:
    """Token ids -> the space-joined sequence string `seq_to_smiles` parses.
    Truncates at the first <eog> (inclusive) if present; a row that never
    emitted <eog> within max_len is passed through as-is and will fail to
    parse (tokens.index('<eog>') raises inside seq_to_mol), which
    correctly maps to the invalid-reward floor rather than a crash --
    `build_reward_smiles`'s callers already catch `Chem.MolFromSmiles is
    None`/exceptions, and `seq_to_smiles` itself returns None on any parse
    failure.

    Bug fixed 2026-09-05: both call sites pass `x[:, 1:].tolist()` (`x`'s
    own first column is always `prior.boc_id`, the seed every rollout
    starts from -- stripped here the same way MolGPT's `_decode` strips its
    PAD seed). But G2PT's grammar is not MolGPT's: `_seq_to_mol` requires a
    literal `<boc>` token (`tokens.index("<boc>")`) to find where the node
    section starts, and with it missing from every row this function ever
    saw, EVERY generated sequence in EVERY guide-training step and every
    final_dump call raised inside `_seq_to_mol` and was scored `syntax_fail`
    -- regardless of what the model actually generated. `token_row` is
    `x`'s row with the seed already stripped, so it's re-added here, once,
    rather than changing either call site (which would have to remember
    to)."""
    toks = [prior.itos[prior.boc_id]] + [prior.itos[i] for i in token_row]
    if "<eog>" in toks:
        toks = toks[:toks.index("<eog>") + 1]
    return " ".join(toks)


class Trainer:
    def __init__(self, cfg: GFNConfig, device: str = "cuda", ckpt_root: str = "logs/g2pt-gfn"):
        self.cfg = cfg
        self.device = device
        self.ckpt_root = Path(ckpt_root)

        self.prior = FrozenG2PT(G2PTConfig(model_name_or_path=cfg.model_name_or_path), device=device)
        d_model = self.prior.n_embd
        # cfg.vocab_size must match the checkpoint's actual padded tensor
        # width, not the 112-entry real vocab -- fail loudly rather than
        # silently building a guide with the wrong output size.
        if cfg.vocab_size != self.prior.padded_vocab_size:
            raise ValueError(
                f"cfg.vocab_size={cfg.vocab_size} but the loaded checkpoint's "
                f"padded vocab is {self.prior.padded_vocab_size}. Set "
                "--vocab_size to match, or leave it at config.py's default "
                "unless you've pointed --model_name_or_path at a different checkpoint.")

        torch.manual_seed(cfg.seed)
        if cfg.use_hidden_guide:
            self.guide = HiddenGuide(
                d_model, self.prior.proj_logits, hidden=cfg.guide_hidden,
                layers=cfg.guide_layers, vocab_size=cfg.vocab_size,
                also_output_residual=cfg.hidden_guide_out_residual,
            ).to(device)
        else:
            self.guide = LogitGuide(d_model, cfg.vocab_size, cfg.guide_hidden, cfg.guide_layers).to(device)

        with torch.no_grad():
            seed = torch.full((2, 1), self.prior.boc_id, dtype=torch.long, device=device)
            h_probe = self.prior.hidden_states(seed)[:, -1, :]
            if isinstance(self.guide, HiddenGuide):
                g0 = self.guide.guided_logits(h_probe)
            else:
                g0 = self.prior.proj_logits(h_probe) + self.guide(h_probe)
            p0 = self.prior.proj_logits(h_probe)
            if not torch.allclose(g0, p0, atol=1e-4):
                raise AssertionError(
                    "guide is not the identity at init -- guided logits differ "
                    f"from prior logits by up to {(g0 - p0).abs().max().item():.4g}.")

        self.flow_head = None
        self.logZ = None
        params = list(self.guide.parameters())
        if cfg.objective == "db":
            self.flow_head = LogFlowHead(d_model, cfg.flow_hidden, cfg.flow_layers).to(device)
            params += list(self.flow_head.parameters())
        elif cfg.objective == "rtb":
            self.logZ = torch.nn.Parameter(torch.tensor(float(cfg.logz_init), device=device))
        else:
            raise ValueError(f"gflow_g2pt.py only implements objective in ('db','rtb'), got {cfg.objective!r}.")

        if cfg.use_replay:
            raise NotImplementedError(
                "Replay is out of scope for this pilot, same reasoning as the MolGPT leg "
                "(gflow_molgpt.py's module docstring) -- no G2PT-shaped replay buffer exists.")
        if cfg.warm_start_guide:
            raise NotImplementedError("warm_start_guide is not wired up here -- leave it unset.")

        opt_groups = [{"params": params, "lr": cfg.lr}]
        if self.logZ is not None:
            opt_groups.append({"params": [self.logZ], "lr": cfg.logz_lr})
        self.opt = torch.optim.AdamW(opt_groups, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.wd)

        # Ported verbatim from gflow.py -- see molgpt/molgpt_gfn/gflow_molgpt.py's
        # module docstring, "Guide EMA, added 2026-09-04" (same reasoning here).
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
        with torch.no_grad():
            h = self.prior.hidden_states(tokens)[:, -1, :]
            prior_logits = self.prior.mask_logits(self.prior.proj_logits(h))
        return h, prior_logits

    def _guided_step(self, guide, prev_token, h, prior_logits, atom_only: bool):
        """Apply Option-A gating: the guide only perturbs atom-decision
        steps (see module docstring). At every other step, `guided ==
        prior_logits` exactly -- not approximately, exactly, since we
        substitute the untouched tensor rather than call the guide and hope
        it learned to be a no-op there. This is what makes the gate a real
        architectural guarantee rather than a training-time tendency.

        Also applies the dimension-wise atom-token mask (author instruction,
        2026-09-04) on top of the step-level gate: even within an
        atom-decision step, the guide may only move logit mass among the
        `ATOM_*` entries, not the (already near-zero-probability there, but
        not architecturally forbidden without this) IDX_*/BOND_*/structural
        entries. Belt-and-suspenders with the step gate above -- see
        `guides.masked_guided_logits_dispatch`'s docstring."""
        guided = masked_guided_logits_dispatch(guide, prior_logits, h, self.prior.atom_token_mask)
        guided = self.prior.mask_logits(guided)
        if not atom_only:
            return guided
        # torch.where's backward pass sends zero gradient into the discarded
        # branch at each masked-out position, so the guide gets exactly zero
        # gradient contribution from non-atom steps (not "near zero" --
        # exactly zero, since prior_logits there is a no_grad tensor). This
        # does still run the guide's full forward pass at every step, even
        # ones whose result gets discarded -- correctness over the extra
        # compute, worth revisiting if profiling ever shows this matters at
        # this pilot's scale.
        is_atom = self.prior.is_atom_step(prev_token)          # [B] bool
        return torch.where(is_atom.unsqueeze(-1), guided, prior_logits)

    def generate_guided(self, bsz, guide, sample_temp=1.0, rand_eps=0.0, max_len=None):
        max_len = max_len or self.cfg.max_len
        device = self.device
        x = torch.full((bsz, 1), self.prior.boc_id, dtype=torch.long, device=device)
        alive = torch.ones(bsz, dtype=torch.bool, device=device)
        logp_policy = torch.zeros(bsz, device=device)
        logp_prior = torch.zeros(bsz, device=device)

        for _t in range(max_len - 1):
            prev_token = x[:, -1]
            h, prior_logits = self._prior_step(x)
            guided = self._guided_step(guide, prev_token, h, prior_logits, self.cfg.atom_guide_only)
            logp_pol = F.log_softmax(guided, dim=-1)
            logp_pri = F.log_softmax(prior_logits, dim=-1)
            behav = F.softmax(guided.detach() / sample_temp, dim=-1)
            if rand_eps > 0:
                behav = (1 - rand_eps) * behav + rand_eps * self.prior.real_token_mask.float() / self.prior.real_token_mask.sum()
            nxt = torch.multinomial(behav, 1).squeeze(-1)

            logp_policy = logp_policy + logp_pol.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) * alive
            logp_prior = logp_prior + logp_pri.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) * alive
            alive = alive & (nxt != self.prior.eog_id)
            x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
            if not alive.any():
                break

        smiles_strs = [_decode(self.prior, row) for row in x[:, 1:].tolist()]
        log_reward = self.compute_log_reward(smiles_strs)
        return {"tokens": x, "seqs": smiles_strs, "logp_policy": logp_policy,
                "logp_prior": logp_prior, "log_reward": log_reward}

    def db_rollout_states(self, guide, bsz, max_len=None):
        max_len = max_len or self.cfg.max_len
        device = self.device
        x = torch.full((bsz, 1), self.prior.boc_id, dtype=torch.long, device=device)
        alive = torch.ones(bsz, dtype=torch.bool, device=device)
        stop_step = torch.full((bsz,), -1, dtype=torch.long, device=device)

        hs_list, logpf_chosen_list, logpf_stop_list = [], [], []
        logprior_step_list, step_mask_list = [], []

        T = max_len - 1
        for t in range(T):
            prev_token = x[:, -1]
            h, prior_logits = self._prior_step(x)
            guided = self._guided_step(guide, prev_token, h, prior_logits, self.cfg.atom_guide_only)
            logp_pol = F.log_softmax(guided, dim=-1)
            with torch.no_grad():
                logp_pri = F.log_softmax(prior_logits, dim=-1)
                behav = F.softmax(guided.detach() / self.cfg.sample_temp, dim=-1)
                if self.cfg.rand_eps > 0:
                    behav = ((1 - self.cfg.rand_eps) * behav
                             + self.cfg.rand_eps * self.prior.real_token_mask.float() / self.prior.real_token_mask.sum())
                nxt = torch.multinomial(behav, 1).squeeze(-1)

            hs_list.append(h)
            logpf_chosen_list.append(logp_pol.gather(-1, nxt.unsqueeze(-1)).squeeze(-1))
            logpf_stop_list.append(logp_pol[:, self.prior.eog_id])
            logprior_step_list.append(logp_pri.gather(-1, nxt.unsqueeze(-1)).squeeze(-1).detach())
            step_mask_list.append(alive.clone())

            just_stopped = alive & (nxt == self.prior.eog_id)
            stop_step = torch.where(just_stopped, torch.full_like(stop_step, t), stop_step)
            alive = alive & (nxt != self.prior.eog_id)
            x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
            if not alive.any():
                for _ in range(T - len(hs_list)):
                    hs_list.append(h)
                    logpf_chosen_list.append(logpf_chosen_list[-1])
                    logpf_stop_list.append(logpf_stop_list[-1])
                    logprior_step_list.append(logprior_step_list[-1])
                    step_mask_list.append(torch.zeros_like(alive))
                break

        stop_step = torch.where(stop_step < 0, torch.full_like(stop_step, T - 1), stop_step)
        smiles_strs = [_decode(self.prior, row) for row in x[:, 1:].tolist()]
        return {
            "hs": torch.stack(hs_list, dim=1),
            "logpf_chosen": torch.stack(logpf_chosen_list, dim=1),
            "logpf_stop": torch.stack(logpf_stop_list, dim=1),
            "logprior_step": torch.stack(logprior_step_list, dim=1),
            "step_mask": torch.stack(step_mask_list, dim=1),
            "stop_step": stop_step,
            "seqs": smiles_strs,
            "tokens": x,
        }

    @staticmethod
    def db_loss_from_rollout(flow_head, roll, log_reward, beta, invalid_logr=None, target_clip=None):
        """Ported verbatim -- identical to gflow_molgpt.py's copy of
        gflow.py's LitGFlowNet.db_loss_from_rollout. See that file for the
        provenance note; not restated here since it's the same code."""
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

    def compute_log_reward(self, seq_strs):
        smiles = [seq_to_smiles(s) or "" for s in seq_strs]
        vals = [self.reward_fn(smi) for smi in smiles]
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    # ---------------------------------------------------------------- training

    def training_step(self):
        cfg = self.cfg
        if cfg.objective == "db":
            roll = self.db_rollout_states(self.guide, cfg.bsz)
            log_reward = self.compute_log_reward(roll["seqs"])
            loss, stats = self.db_loss_from_rollout(
                self.flow_head, roll, log_reward, cfg.reward_beta,
                invalid_logr=cfg.invalid_logr, target_clip=cfg.db_target_clip)
        else:  # rtb
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
        self.guide_ema.update_parameters(self.guide)

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
                project="g2pt-gfn", entity=os.getenv("WANDB_ENTITY"),
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
        if isinstance(f.default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(flag, type=type(f.default) if f.default is not None else str, default=f.default)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt_root", default="logs/g2pt-gfn")
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
