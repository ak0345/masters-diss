"""Flip diagnostics for a trained MolGPT guide.

Adapted from `ablations/single_flip_ablation.py` -- same diagnostic
*definitions* (delivered_frac, argmax_flip_rate, sample_flip_rate,
mean_total_variation, mean_KL, mean_prior_top1_gap, flip_rate_by_position),
ported for a token-only autoregressive prior. No coordinate diffusion, no
atomic-number mask to apply before softmax -- MolGPT's full 94-token vocab is
legal at every position.

**Trajectories belong to the prior.** The rollout below is driven entirely by
the frozen prior's own sampled tokens (`next_prior`); the guide's
distribution is computed at every visited state purely to measure it, never
to steer. This is what makes every guide architecture comparable on the same
state distribution (see plan_molgpt.md, "Diagnostics"). A shared uniform draw
`u` per step samples both `next_prior` and the guide's hypothetical
`next_g`, so `sample_flip_rate` isolates the guide's effect from independent
sampling noise.

Usage: `python flip_molgpt.py --ckpt <guide checkpoint> --label <name> --out_dir results/molgpt/flips`
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
from molgpt_prior import FrozenMolGPT, MolGPTConfig  # noqa: E402
from guides import HiddenGuide, LogitGuide, masked_guided_logits_dispatch  # noqa: E402

N_REPORT_POS = 256
# Added 2026-09-06, author instruction: a cross-model positional comparison
# needs a common x-axis unit. Raw SMILES-token position isn't one -- MolGPT
# has no step-level atom/non-atom gate the way G2PT does (see guides.py's
# masked_guided_logits_dispatch docstring: no clean "next token will be an
# atom" rule exists for SMILES the way it does for G2PT's rigid grammar), so
# whether a given step IS an atom decision can only be read off the REALIZED
# token, per-trajectory, after the fact -- not predicted in lockstep across
# the batch the way G2PT's `t // 3` is. `flip_rate_by_atom_index` below
# buckets by "how many atoms this trajectory has already placed", which is
# the unit comparable to Quetzal's native atom-index and to G2PT's
# raw-position-integer-divided-by-3 (both already atom-native or a fixed
# stride away from it).
MAX_ATOMS_BUCKET = 150


def load_guide_checkpoint(ckpt_path, prior: FrozenMolGPT, device, guide_source: str = "ema"):
    """Load a guide checkpoint written by `gflow_molgpt.Trainer.save_checkpoint`.

    `guide_source`: "ema" (default, matches `final_dump.py`'s own default and
    is what every published Quetzal flip-diagnostic number in
    `results/flips-guide/_aggs/` used -- see `papers/current/decisions.md`,
    2026-09-04) or "policy" (the raw trained weights, pre-EMA).

    Same fail-loudly discipline as `final_dump.py`'s checkpoint-unwrap logic
    (plan_molgpt.md, "Genuinely new"): a guide-type mismatch or missing guide
    weights raises immediately rather than silently reporting numbers for a
    guide that never left initialisation."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GFNConfig(**ckpt["config"])
    d_model = prior.cfg.n_embd

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
            f"but checkpoint's guide_class is {guide_class!r}. Config did not "
            "round-trip; refusing to score a mismatched guide.")

    if guide_source == "ema":
        sd_key = "guide_ema_state_dict"
    elif guide_source == "policy":
        sd_key = "guide_state_dict"
    else:
        raise ValueError(f"guide_source must be 'ema' or 'policy', got {guide_source!r}")
    if sd_key not in ckpt:
        raise SystemExit(
            f"[FATAL] checkpoint {ckpt_path} has no {sd_key!r} -- it predates the "
            "2026-09-04 EMA addition (see papers/current/decisions.md) and must "
            "be retrained, not scored as if it had an EMA guide.")

    missing, _unexpected = guide.load_state_dict(ckpt[sd_key], strict=False)
    if missing:
        raise SystemExit(
            f"[FATAL] {len(missing)} guide weights did not load from {ckpt_path} "
            f"(e.g. {missing[:4]}). The guide is untrained -> guided would equal "
            "the prior. Aborting rather than reporting a false negative.")

    guide.to(device).eval()
    for p in guide.parameters():
        p.requires_grad_(False)
    return guide, cfg, ckpt.get("step"), ckpt.get("epoch")


def _pos_rates(numerator_by_pos, denominator_by_pos, n_report=N_REPORT_POS):
    """Divide elementwise, but positions with a zero denominator (no
    trajectory reached them -- every one had already stopped) come back as
    JSON `null`, never `0.0`. This is the exact mechanism from
    ablations/single_flip_ablation.py's `_pos_rates`: pre-fill with NaN, only
    overwrite where the denominator is positive, then convert NaN -> None for
    JSON serialisation. A position with real state_count==0 must never look
    like a position where nothing flipped."""
    n = int(min(n_report, len(numerator_by_pos)))
    num = np.asarray(numerator_by_pos[:n], dtype=float)
    den = np.asarray(denominator_by_pos[:n], dtype=float)
    rates = np.divide(num, den, out=np.full(n, np.nan), where=den > 0)
    return ([None if np.isnan(r) else float(r) for r in rates],
            [int(s) for s in den])


@torch.no_grad()
def flip_diagnostics(prior: FrozenMolGPT, guide, n_traj: int, temp: float,
                      max_len: int, device: str, seed=None) -> dict:
    if seed is not None:
        torch.manual_seed(seed)
    bsz = n_traj
    x = torch.full((bsz, 1), prior.seed_idx, dtype=torch.long, device=device)
    alive = torch.ones(bsz, dtype=torch.bool, device=device)

    acc = {"n_states": 0, "delivered": 0, "argmax_flip": 0, "sample_flip": 0,
           "mass_moved_sum": 0.0, "kl_sum": 0.0, "prior_top1_logit_gap_sum": 0.0}
    T = max_len - 1
    flip_by_pos = [0] * T
    state_by_pos = [0] * T
    gap_sum_by_pos = [0.0] * T

    # Atom-decision-index accumulators (see MAX_ATOMS_BUCKET's comment above).
    # atom_count[i] = how many atom tokens trajectory i has actually realized
    # so far -- updated below from the PRIOR's realized draw, same rollout
    # `x` already advances on, so this is well-defined regardless of what the
    # guide would have done.
    atom_count = torch.zeros(bsz, dtype=torch.long, device=device)
    flip_by_atom = torch.zeros(MAX_ATOMS_BUCKET, device=device)
    state_by_atom = torch.zeros(MAX_ATOMS_BUCKET, device=device)
    gap_sum_by_atom = torch.zeros(MAX_ATOMS_BUCKET, device=device)

    for t in range(T):
        h = prior.hidden_states(x)[:, -1, :]
        prior_logits = prior.proj_logits(h)
        guided_logits = masked_guided_logits_dispatch(guide, prior_logits, h, prior.atom_token_mask)

        # temp scales the softmax-derived quantities (probabilities, sampling,
        # TV/KL) identically for prior and guided, so comparisons at a given
        # temperature regime stay apples-to-apples; argmax and the raw-logit
        # `delivered` check are temperature-invariant and left alone.
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
        acc["n_states"] += int(alive.sum().item())
        acc["delivered"] += int((delivered & alive).sum().item())
        acc["argmax_flip"] += int((argmax_flip & alive).sum().item())
        acc["sample_flip"] += int((sample_flip & alive).sum().item())
        acc["mass_moved_sum"] += float((mass_moved * alive_f).sum().item())
        acc["kl_sum"] += float((kl * alive_f).sum().item())
        acc["prior_top1_logit_gap_sum"] += float((gap * alive_f).sum().item())

        flip_by_pos[t] += int((sample_flip & alive).sum().item())
        state_by_pos[t] += int(alive.sum().item())
        gap_sum_by_pos[t] += float((gap * alive_f).sum().item())

        # Bucket ONLY steps whose REALIZED token is an atom -- MolGPT has no
        # step-level pre-check the way G2PT's `is_atom` is (see this file's
        # MAX_ATOMS_BUCKET comment), so "is this an atom decision" can only
        # be read off next_prior, already drawn above. Bucketing every step
        # regardless would dilute the atom-decision signal with the many
        # non-atom SMILES-syntax steps around it, the same reasoning as
        # flip_g2pt.py's equivalent fix. atom_count is still pre-increment
        # here, so it's the correct 0-based index of THIS atom decision.
        is_atom_realized = prior.atom_token_mask[next_prior]
        m = alive & is_atom_realized
        ac = atom_count.clamp(max=MAX_ATOMS_BUCKET - 1)[m]
        if ac.numel():
            flip_by_atom.scatter_add_(0, ac, sample_flip[m].float())
            state_by_atom.scatter_add_(0, ac, torch.ones_like(ac, dtype=torch.float))
            gap_sum_by_atom.scatter_add_(0, ac, gap[m])

        # Advance with the PRIOR's draw, always -- the guide never steers.
        x = torch.cat([x, next_prior.unsqueeze(1)], dim=1)
        atom_count = atom_count + is_atom_realized.long()
        alive = alive & (next_prior != prior.pad_idx)
        if not alive.any():
            break

    ns = max(acc["n_states"], 1)
    flip_rate_by_position, state_count_by_position = _pos_rates(flip_by_pos, state_by_pos)
    mean_gap_by_position, _ = _pos_rates(gap_sum_by_pos, state_by_pos)
    flip_by_atom_l = flip_by_atom.cpu().tolist()
    state_by_atom_l = state_by_atom.cpu().tolist()
    gap_sum_by_atom_l = gap_sum_by_atom.cpu().tolist()
    flip_rate_by_atom_index, state_count_by_atom_index = _pos_rates(flip_by_atom_l, state_by_atom_l)
    mean_gap_by_atom_index, _ = _pos_rates(gap_sum_by_atom_l, state_by_atom_l)

    return {
        "n_trajectories": n_traj,
        "n_states": acc["n_states"],
        "delivered_frac": acc["delivered"] / ns,
        "argmax_flip_rate": acc["argmax_flip"] / ns,
        "sample_flip_rate": acc["sample_flip"] / ns,
        "mean_total_variation": acc["mass_moved_sum"] / ns,
        "mean_KL": acc["kl_sum"] / ns,
        "mean_prior_top1_gap": acc["prior_top1_logit_gap_sum"] / ns,
        "flip_rate_by_position": flip_rate_by_position,
        "state_count_by_position": state_count_by_position,
        "mean_gap_by_position": mean_gap_by_position,
        # Atom-decision-index versions of the same three -- see
        # MAX_ATOMS_BUCKET's comment: the unit comparable across models.
        "flip_rate_by_atom_index": flip_rate_by_atom_index,
        "state_count_by_atom_index": state_count_by_atom_index,
        "mean_gap_by_atom_index": mean_gap_by_atom_index,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="guide checkpoint from gflow_molgpt.py")
    ap.add_argument("--molgpt_ckpt", default="checkpoints/molgpt_geom_drugs_unconditional.pt",
                     help="frozen MolGPT prior checkpoint (independent of --ckpt)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--report_tag", default=None, help="filename-safe tag; defaults to --label")
    ap.add_argument("--out_dir", default="results/molgpt/flips")
    ap.add_argument("--flip_temp", type=float, default=1.0)
    ap.add_argument("--also_temp", type=float, default=None)
    ap.add_argument("--n_traj", type=int, default=2000)
    ap.add_argument("--n_report_pos", type=int, default=N_REPORT_POS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--guide_source", choices=["ema", "policy"], default="ema",
                     help="matches final_dump.py's default -- see load_guide_checkpoint's docstring")
    args = ap.parse_args()

    tag = args.report_tag or "".join(c if c.isalnum() or c in "-_." else "_" for c in args.label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"flip_report_{tag}.json"

    molcfg = MolGPTConfig()
    prior = FrozenMolGPT(molcfg, args.molgpt_ckpt, device=args.device)

    guide, guide_cfg, step, epoch = load_guide_checkpoint(args.ckpt, prior, args.device, args.guide_source)

    temps = [args.flip_temp] + ([args.also_temp] if args.also_temp else [])
    results = {}
    for temp in temps:
        results[f"t{temp}"] = flip_diagnostics(
            prior, guide, args.n_traj, temp, molcfg.block_size, args.device, seed=args.seed)

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
        print(f"  {temp}: delivered={r['delivered_frac']:.3f} "
              f"argmax_flip={r['argmax_flip_rate']:.3f} "
              f"sample_flip={r['sample_flip_rate']:.3f} "
              f"mean_TV={r['mean_total_variation']:.4f}")


if __name__ == "__main__":
    main()
