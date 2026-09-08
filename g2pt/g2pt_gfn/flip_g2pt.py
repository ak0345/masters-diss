"""Flip diagnostics for a trained G2PT guide.

Adapted from `molgpt/molgpt_gfn/flip_molgpt.py` -- same diagnostic
*definitions* (delivered_frac, argmax_flip_rate, sample_flip_rate,
mean_total_variation, mean_KL, mean_prior_top1_gap, flip_rate_by_position),
same "trajectories belong to the prior" design (the guide is measured, never
used to steer the rollout).

**The one thing genuinely new here**: because this leg's guide is Option A
(atom-decision-only, gated in the rollout builder rather than by the model's
own architecture -- see `papers/current/decisions.md`, "G2PT leg uses Option
A"), every metric is reported three ways: `overall` (every step, matching
the other legs' single number), `atom_steps` (only steps where
`FrozenG2PT.is_atom_step` is true -- this is where delivery should read
~1.0, the actual analogue of the other legs' single delivered_frac), and
`non_atom_steps` (IDX_*/BOND_*/structural steps -- delivered_frac here
should read EXACTLY 0.0, not "small", since the gate substitutes the raw
prior logits verbatim at those positions; if it doesn't read exactly 0, the
gate has a bug, not a finding).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GFNConfig  # noqa: E402
from g2pt_prior import FrozenG2PT, G2PTConfig  # noqa: E402
from guides import HiddenGuide, LogitGuide, masked_guided_logits_dispatch  # noqa: E402

N_REPORT_POS = 256


def load_guide_checkpoint(ckpt_path, prior: FrozenG2PT, device, guide_source: str = "ema"):
    """Same fail-loudly discipline as flip_molgpt.py's loader, including the
    `guide_source` default -- see that file's docstring for why 'ema' is the
    default (matches every published Quetzal flip-diagnostic number)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GFNConfig(**ckpt["config"])
    d_model = prior.n_embd

    guide_class = ckpt.get("guide_class")
    if guide_class == "HiddenGuide":
        guide = HiddenGuide(d_model, prior.proj_logits, hidden=cfg.guide_hidden,
                             layers=cfg.guide_layers, vocab_size=cfg.vocab_size,
                             also_output_residual=cfg.hidden_guide_out_residual)
    elif guide_class == "LogitGuide":
        guide = LogitGuide(d_model, cfg.vocab_size, cfg.guide_hidden, cfg.guide_layers)
    else:
        raise SystemExit(f"[FATAL] unknown guide_class {guide_class!r} in checkpoint {ckpt_path}")

    expected_hidden = bool(cfg.use_hidden_guide)
    built_hidden = isinstance(guide, HiddenGuide)
    if expected_hidden != built_hidden:
        raise SystemExit(
            f"[FATAL] guide-type mismatch: config.use_hidden_guide={expected_hidden} "
            f"but checkpoint's guide_class is {guide_class!r}.")

    if guide_source == "ema":
        sd_key = "guide_ema_state_dict"
    elif guide_source == "policy":
        sd_key = "guide_state_dict"
    else:
        raise ValueError(f"guide_source must be 'ema' or 'policy', got {guide_source!r}")
    if sd_key not in ckpt:
        raise SystemExit(
            f"[FATAL] checkpoint {ckpt_path} has no {sd_key!r} -- it predates the "
            "2026-09-04 EMA addition and must be retrained.")

    missing, _unexpected = guide.load_state_dict(ckpt[sd_key], strict=False)
    if missing:
        raise SystemExit(
            f"[FATAL] {len(missing)} guide weights did not load from {ckpt_path} "
            f"(e.g. {missing[:4]}). The guide is untrained.")

    guide.to(device).eval()
    for p in guide.parameters():
        p.requires_grad_(False)
    return guide, cfg, ckpt.get("step"), ckpt.get("epoch")


def _pos_rates(numerator_by_pos, denominator_by_pos, n_report=N_REPORT_POS):
    n = int(min(n_report, len(numerator_by_pos)))
    num = np.asarray(numerator_by_pos[:n], dtype=float)
    den = np.asarray(denominator_by_pos[:n], dtype=float)
    rates = np.divide(num, den, out=np.full(n, np.nan), where=den > 0)
    return ([None if np.isnan(r) else float(r) for r in rates],
            [int(s) for s in den])


@torch.no_grad()
def flip_diagnostics(prior: FrozenG2PT, guide, n_traj: int, temp: float,
                      max_len: int, device: str, atom_guide_only: bool, seed=None) -> dict:
    if seed is not None:
        torch.manual_seed(seed)
    bsz = n_traj
    x = torch.full((bsz, 1), prior.boc_id, dtype=torch.long, device=device)
    alive = torch.ones(bsz, dtype=torch.bool, device=device)

    # Separate accumulators for atom-decision steps vs everything else, plus
    # a combined "overall" pass -- see module docstring.
    def new_acc():
        return {"n_states": 0, "delivered": 0, "argmax_flip": 0, "sample_flip": 0,
                "mass_moved_sum": 0.0, "kl_sum": 0.0, "prior_top1_logit_gap_sum": 0.0}
    accs = {"overall": new_acc(), "atom_steps": new_acc(), "non_atom_steps": new_acc()}

    T = max_len - 1
    flip_by_pos = [0] * T
    state_by_pos = [0] * T
    gap_sum_by_pos = [0.0] * T

    # Atom-decision-index accumulators, added 2026-09-06 (author instruction):
    # a cross-model positional comparison needs a common x-axis unit, and raw
    # sequence position isn't one -- here it isn't even safe to assume a
    # fixed stride (position // 3): that holds only while every trajectory is
    # still inside the node-construction section, and different molecules
    # exit into the edge-list section (where is_atom is False for the rest
    # of the trajectory) at different raw positions depending on molecule
    # size, so a fixed global stride would misattribute pooled data at later
    # positions where trajectories are in different grammar sections. This
    # tracks each trajectory's own atom-decision count instead, exactly
    # mirroring the equivalent fix in flip_molgpt.py (which needs a realized-
    # token check instead of `is_atom` since MolGPT has no step-level gate).
    atom_count = torch.zeros(bsz, dtype=torch.long, device=device)
    MAX_ATOMS_BUCKET = 150
    flip_by_atom = torch.zeros(MAX_ATOMS_BUCKET, device=device)
    state_by_atom = torch.zeros(MAX_ATOMS_BUCKET, device=device)
    gap_sum_by_atom = torch.zeros(MAX_ATOMS_BUCKET, device=device)

    for t in range(T):
        prev_token = x[:, -1]
        h = prior.hidden_states(x)[:, -1, :]
        prior_logits = prior.mask_logits(prior.proj_logits(h))
        guided_logits = prior.mask_logits(masked_guided_logits_dispatch(guide, prior_logits, h, prior.atom_token_mask))
        is_atom = prior.is_atom_step(prev_token)  # [B] bool -- which regime each row is in

        if atom_guide_only:
            guided_logits = torch.where(is_atom.unsqueeze(-1), guided_logits, prior_logits)

        p_prior = F.softmax(prior_logits / temp, dim=-1)
        p_g = F.softmax(guided_logits / temp, dim=-1)
        lp_prior = F.log_softmax(prior_logits / temp, dim=-1)
        lp_g = F.log_softmax(guided_logits / temp, dim=-1)

        delivered = (guided_logits - prior_logits).abs().sum(-1) > 1e-6
        prior_top1 = prior_logits.argmax(-1)
        g_top1 = guided_logits.argmax(-1)
        argmax_flip = g_top1 != prior_top1

        u = torch.rand(bsz, 1, device=device)
        next_prior = (u < torch.cumsum(p_prior, dim=-1)).float().argmax(dim=-1)
        next_g = (u < torch.cumsum(p_g, dim=-1)).float().argmax(dim=-1)
        sample_flip = next_g != next_prior

        mass_moved = 0.5 * (p_g - p_prior).abs().sum(-1)
        kl = (p_g * (lp_g - lp_prior)).sum(-1)
        top2 = torch.topk(prior_logits, 2, dim=-1).values
        gap = top2[:, 0] - top2[:, 1]

        alive_f = alive.float()
        for regime_name, regime_mask in (("overall", torch.ones_like(alive)),
                                          ("atom_steps", is_atom),
                                          ("non_atom_steps", ~is_atom)):
            m = (alive & regime_mask)
            mf = m.float()
            acc = accs[regime_name]
            acc["n_states"] += int(m.sum().item())
            acc["delivered"] += int((delivered & m).sum().item())
            acc["argmax_flip"] += int((argmax_flip & m).sum().item())
            acc["sample_flip"] += int((sample_flip & m).sum().item())
            acc["mass_moved_sum"] += float((mass_moved * mf).sum().item())
            acc["kl_sum"] += float((kl * mf).sum().item())
            acc["prior_top1_logit_gap_sum"] += float((gap * mf).sum().item())

        flip_by_pos[t] += int((sample_flip & alive).sum().item())
        state_by_pos[t] += int(alive.sum().item())
        gap_sum_by_pos[t] += float((gap * alive_f).sum().item())

        # Bucket ONLY the atom-decision steps themselves -- bucketing every
        # step (atom and non-atom alike) by the running count would pool
        # each atom decision's signal together with the guaranteed-zero
        # non-atom steps immediately around it (is_atom is False there by
        # construction), diluting the true atom-decision flip rate rather
        # than isolating it. `atom_count` here is still pre-increment, so
        # it's the correct 0-based index of THIS decision.
        m = alive & is_atom
        ac = atom_count.clamp(max=MAX_ATOMS_BUCKET - 1)[m]
        if ac.numel():
            flip_by_atom.scatter_add_(0, ac, sample_flip[m].float())
            state_by_atom.scatter_add_(0, ac, torch.ones_like(ac, dtype=torch.float))
            gap_sum_by_atom.scatter_add_(0, ac, gap[m])

        x = torch.cat([x, next_prior.unsqueeze(1)], dim=1)
        atom_count = atom_count + is_atom.long()
        alive = alive & (next_prior != prior.eog_id)
        if not alive.any():
            break

    flip_rate_by_position, state_count_by_position = _pos_rates(flip_by_pos, state_by_pos)
    mean_gap_by_position, _ = _pos_rates(gap_sum_by_pos, state_by_pos)
    flip_by_atom_l = flip_by_atom.cpu().tolist()
    state_by_atom_l = state_by_atom.cpu().tolist()
    gap_sum_by_atom_l = gap_sum_by_atom.cpu().tolist()
    flip_rate_by_atom_index, state_count_by_atom_index = _pos_rates(flip_by_atom_l, state_by_atom_l)
    mean_gap_by_atom_index, _ = _pos_rates(gap_sum_by_atom_l, state_by_atom_l)

    results = {}
    for regime_name, acc in accs.items():
        ns = max(acc["n_states"], 1)
        results[regime_name] = {
            "n_states": acc["n_states"],
            "delivered_frac": acc["delivered"] / ns,
            "argmax_flip_rate": acc["argmax_flip"] / ns,
            "sample_flip_rate": acc["sample_flip"] / ns,
            "mean_total_variation": acc["mass_moved_sum"] / ns,
            "mean_KL": acc["kl_sum"] / ns,
            "mean_prior_top1_gap": acc["prior_top1_logit_gap_sum"] / ns,
        }
    results["overall"]["flip_rate_by_position"] = flip_rate_by_position
    results["overall"]["state_count_by_position"] = state_count_by_position
    results["overall"]["mean_gap_by_position"] = mean_gap_by_position
    # Atom-decision-index versions -- the unit comparable across models (see
    # MAX_ATOMS_BUCKET's comment above).
    results["overall"]["flip_rate_by_atom_index"] = flip_rate_by_atom_index
    results["overall"]["state_count_by_atom_index"] = state_count_by_atom_index
    results["overall"]["mean_gap_by_atom_index"] = mean_gap_by_atom_index
    return {"n_trajectories": n_traj, "by_regime": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name_or_path", default="checkpoints/g2pt_geom_drugs_hf")
    ap.add_argument("--label", required=True)
    ap.add_argument("--report_tag", default=None)
    ap.add_argument("--out_dir", default="results/g2pt/flips")
    ap.add_argument("--flip_temp", type=float, default=1.0)
    ap.add_argument("--also_temp", type=float, default=None)
    ap.add_argument("--n_traj", type=int, default=500)
    ap.add_argument("--n_report_pos", type=int, default=N_REPORT_POS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--guide_source", choices=["ema", "policy"], default="ema")
    args = ap.parse_args()

    tag = args.report_tag or "".join(c if c.isalnum() or c in "-_." else "_" for c in args.label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"flip_report_{tag}.json"

    prior = FrozenG2PT(G2PTConfig(model_name_or_path=args.model_name_or_path), device=args.device)
    guide, guide_cfg, step, epoch = load_guide_checkpoint(args.ckpt, prior, args.device, args.guide_source)

    temps = [args.flip_temp] + ([args.also_temp] if args.also_temp else [])
    results = {}
    for temp in temps:
        results[f"t{temp}"] = flip_diagnostics(
            prior, guide, args.n_traj, temp, prior.block_size, args.device,
            atom_guide_only=guide_cfg.atom_guide_only, seed=args.seed)

    report = {
        "label": args.label,
        "ckpt": str(args.ckpt),
        "guide_source": args.guide_source,
        "guide_class": type(guide).__name__,
        "guide_config": asdict(guide_cfg),
        "train_step": step,
        "train_epoch": epoch,
        "n_traj": args.n_traj,
        "results_by_temp": results,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    for temp, r in results.items():
        for regime in ("atom_steps", "non_atom_steps"):
            reg = r["by_regime"][regime]
            print(f"  {temp} [{regime}]: n={reg['n_states']} "
                  f"delivered={reg['delivered_frac']:.3f} "
                  f"argmax_flip={reg['argmax_flip_rate']:.3f} "
                  f"sample_flip={reg['sample_flip_rate']:.3f} "
                  f"mean_TV={reg['mean_total_variation']:.4f}")


if __name__ == "__main__":
    main()
