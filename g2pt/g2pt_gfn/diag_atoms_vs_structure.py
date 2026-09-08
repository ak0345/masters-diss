#!/usr/bin/env python3
"""Where does the reward live for G2PT: in the node tokens a guide can change,
or in the edge tokens it cannot?

This is the G2PT half of the pilot analogue to Quetzal's atoms-vs-coordinates ablation. See
`molgpt/molgpt_gfn/diag_atoms_vs_structure.py` for the shared rationale.

G2PT's grammar interleaves node and edge tokens, and the guide is gated to atom-decision steps
by `atom_guide_only`. The structural channel here is therefore the edge tokens, which is a
cleaner analogue of Quetzal's coordinates than MolGPT's bond and ring-closure characters: an
edge can be resampled without necessarily breaking the string.

Original shared docstring follows.

Where does the reward live for a 1D/2D prior: in the tokens a guide can change,
or in the ones it cannot?

Quetzal's `ablations/diag_atoms_vs_coords.py` asks whether the MPO reward is carried by the
atom types a guide edits or by the 3D coordinates the frozen diffusion produces. MolGPT and
G2PT have no coordinate channel, so that exact test does not transfer. They have the same
problem in a different shape.

Both priors emit a single token stream mixing two kinds of token:

    atom tokens        element symbols and bracket atoms      -- the guide acts here
    structural tokens  bonds, ring closures, branches (MolGPT)
                       edge tokens (G2PT)                     -- the guide never acts here

A reward carried mostly by the structural tokens would be out of reach of an atom-level guide
on these architectures for the same reason geometry is out of reach on Quetzal, and a flat
result against it would say nothing about steering.

Two tests, both on frozen-prior molecules with no further training:

  T1 ATOM SENSITIVITY       roll out from the prior, then re-roll forcing every atom-token
                            decision to the guide's top-1 choice while leaving structural
                            decisions to the prior's own draw. A large reward change means
                            atom tokens are a lever the guide can pull.

  T2 STRUCTURE SENSITIVITY  hold the realised atom tokens fixed and re-roll only the
                            structural decisions, several times, rescoring each. High reward
                            variance from structure alone means the objective is carried
                            where the guide cannot reach.

T2 is the direct analogue of re-rolling Quetzal's coordinate diffusion with atoms held fixed.
Unlike coordinates, a resampled structural token can produce an unparseable string, so the
invalid fraction is reported alongside the variance rather than being silently dropped: a
structure channel that cannot be perturbed without breaking the molecule is itself a finding.

Usage:
    python g2pt_gfn/diag_atoms_vs_structure.py --ckpt <guide.ckpt> --reward osim
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GFNConfig                                        # noqa: E402
from g2pt_prior import FrozenG2PT, G2PTConfig, seq_to_smiles       # noqa: E402
from gflow_g2pt import _decode                                      # noqa: E402
from flip_g2pt import load_guide_checkpoint                         # noqa: E402
from reward_adapter import build_reward_smiles                      # noqa: E402

LEG = "g2pt"


def rollout(prior, guide, n, max_len, device, force_atom_top1=False,
            fixed_atoms=None, temp=1.0, seed=None):
    """One batched rollout.

    force_atom_top1  at atom-token steps, take the guide's top-1 instead of sampling.
    fixed_atoms      list per trajectory of atom tokens to replay in order; structural
                     steps are sampled freshly. This is the structure re-roll.
    """
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.full((n, 1), prior.boc_id, dtype=torch.long, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    atom_cursor = torch.zeros(n, dtype=torch.long, device=device)
    realised_atoms = [[] for _ in range(n)]

    for _ in range(max_len - 1):
        if not bool(alive.any()):
            break
        prev_token = x[:, -1]
        h = prior.hidden_states(x)[:, -1, :]
        logits = prior.mask_logits(prior.proj_logits(h))
        probs = F.softmax(logits / temp, dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(-1)

        # The guide is gated to atom-DECISION steps, which the grammar fixes from the
        # previous token, not from whichever token this step happened to draw. Use the
        # same predicate `gflow_g2pt.Trainer._guided_step` gates on.
        is_atom = prior.is_atom_step(prev_token)

        if force_atom_top1 and guide is not None:
            from flip_g2pt import masked_guided_logits_dispatch
            g_logits = masked_guided_logits_dispatch(guide, logits, h, prior.atom_token_mask)
            g_top1 = (g_logits.masked_fill(~prior.atom_token_mask, -1e30)).argmax(-1)
            nxt = torch.where(is_atom, g_top1, nxt)

        if fixed_atoms is not None:
            replay = torch.full_like(nxt, -1)
            for i in range(n):
                if is_atom[i] and atom_cursor[i] < len(fixed_atoms[i]):
                    replay[i] = fixed_atoms[i][atom_cursor[i]]
            use = replay >= 0
            nxt = torch.where(use, replay.clamp(min=0), nxt)

        for i in range(n):
            if alive[i] and bool(is_atom[i]):
                realised_atoms[i].append(int(nxt[i]))
                atom_cursor[i] += 1

        x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
        alive = alive & (nxt != prior.eog_id)

    # `_decode` re-adds the `<boc>` seed itself, so hand it the row with the seed
    # column stripped, exactly as `gflow_g2pt` does (`x[:, 1:].tolist()`).
    smis = [_decode(prior, row[1:].tolist()) for row in x]
    return smis, realised_atoms


def rollout_batched(prior, guide, n, max_len, device, batch=150, seed=None,
                    fixed_atoms=None, **kw):
    """`rollout` in fixed-size batches, so a large n does not have to fit in one
    attention pass. G2PT parses only about a fifth of what it emits, so n has to be
    several hundred before the surviving sample is worth a mean, and a single pass at
    that size exhausts a shared GPU.

    The seed is set once and the stream runs on across batches, so a given (n, batch)
    is reproducible and the T1 base and forced rollouts stay paired: both walk the same
    batch boundaries with the same starting seed."""
    if seed is not None:
        torch.manual_seed(seed)
    smis, atoms = [], []
    for start in range(0, n, batch):
        m = min(batch, n - start)
        fa = fixed_atoms[start:start + m] if fixed_atoms is not None else None
        s, a = rollout(prior, guide, m, max_len, device, fixed_atoms=fa, seed=None, **kw)
        smis += s
        atoms += a
    return smis, atoms


def score(seqs, log_reward):
    """`rollout` returns G2PT grammar sequences, not SMILES, and the reward takes
    SMILES. `Trainer.compute_log_reward` converts first; so does this. Scoring the
    sequence string directly sends every molecule to the invalid floor.

    Unparseable sequences come back as NaN rather than at `cfg.invalid_logr`, matching
    `molgpt_gfn/diag_atoms_vs_structure.py`. See that file's `score` for why."""
    out = []
    for s in seqs:
        try:
            smi = seq_to_smiles(s)
            ok = bool(smi) and Chem.MolFromSmiles(smi) is not None
            out.append(float(log_reward(smi)) if ok else float("nan"))
        except Exception:
            out.append(float("nan"))
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reward", default="osim", choices=["osim", "peri", "fexo", "nitrogen"])
    ap.add_argument("--n_t1", type=int, default=200)
    ap.add_argument("--n_t2", type=int, default=150)
    ap.add_argument("--rerolls", type=int, default=5)
    ap.add_argument("--batch", type=int, default=150,
                    help="rollout batch size; large n needs this on a shared GPU")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_name_or_path",
                    default="checkpoints/g2pt_geom_drugs_hf")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    BENCH = {"osim": ("guacamol", "hard_osimertinib"), "peri": ("guacamol", "perindopril_rings"),
             "fexo": ("guacamol", "hard_fexofenadine"), "nitrogen": ("nitrogen_count", None)}
    kind, smiles_key = BENCH[args.reward]
    cfg = GFNConfig(reward=kind, reward_smiles=smiles_key)
    prior = FrozenG2PT(G2PTConfig(model_name_or_path=args.model_name_or_path),
                       device=args.device)
    guide, _, _, _ = load_guide_checkpoint(args.ckpt, prior, args.device)
    log_reward = build_reward_smiles(cfg)

    res = {"leg": LEG, "reward": args.reward, "ckpt": args.ckpt}

    # ---- T1: atom sensitivity
    base_smis, atoms = rollout_batched(prior, guide, args.n_t1, cfg.max_len, args.device,
                                       batch=args.batch, seed=0)
    forced_smis, _ = rollout_batched(prior, guide, args.n_t1, cfg.max_len, args.device,
                                     batch=args.batch, seed=0, force_atom_top1=True)
    rb, rf = score(base_smis, log_reward), score(forced_smis, log_reward)
    ok = np.isfinite(rb) & np.isfinite(rf)
    d1 = np.abs(rf[ok] - rb[ok])
    res["T1_atom_sensitivity"] = {
        "n": int(ok.sum()),
        "base_mean": float(rb[ok].mean()), "forced_mean": float(rf[ok].mean()),
        "base_median": float(np.median(rb[ok])), "forced_median": float(np.median(rf[ok])),
        "mean_abs_change": float(d1.mean()),
        # The reward is log(score) with no floor for a valid molecule, so a GuacaMol score
        # near zero reaches -14 or below and a handful of them dominate the mean. The median
        # is the statistic to quote where the two disagree; both are reported so the
        # disagreement itself is visible.
        "median_abs_change": float(np.median(d1)),
        "q90_abs_change": float(np.quantile(d1, 0.90)),
        "base_valid_frac": float(np.isfinite(rb).mean()),
        "forced_valid_frac": float(np.isfinite(rf).mean()),
        "paired_base_rewards": [float(v) for v in rb[ok]],
        "paired_forced_rewards": [float(v) for v in rf[ok]],
    }

    # ---- T2: structure sensitivity, atoms held fixed
    base2, atoms2 = rollout_batched(prior, guide, args.n_t2, cfg.max_len, args.device,
                                    batch=args.batch, seed=1)
    r0 = score(base2, log_reward)
    rolls = []
    for k in range(args.rerolls):
        smis, _ = rollout_batched(prior, guide, args.n_t2, cfg.max_len, args.device,
                                  batch=args.batch, fixed_atoms=atoms2, seed=100 + k)
        rolls.append(score(smis, log_reward))
    R = np.vstack(rolls)
    finite = np.isfinite(R)
    per_mol_std = np.array([R[finite[:, j], j].std() for j in range(R.shape[1])
                            if finite[:, j].sum() >= 2])
    res["T2_structure_sensitivity"] = {
        "n_molecules": int(len(per_mol_std)), "rerolls": args.rerolls,
        "mean_std_from_structure": float(per_mol_std.mean()),
        "median_std_from_structure": float(np.median(per_mol_std)),
        "q90_std_from_structure": float(np.quantile(per_mol_std, 0.90)),
        "per_molecule_std": [float(v) for v in per_mol_std],
        "reroll_valid_frac": float(finite.mean()),
        "base_valid_frac": float(np.isfinite(r0).mean()),
    }

    out = args.out or os.path.join("results", "ablations", "atoms-vs-structure",
                                   "%s_%s.json" % (LEG, args.reward))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, sort_keys=True)
    print(json.dumps(res, indent=2, sort_keys=True))
    print("wrote", out)


if __name__ == "__main__":
    main()
