"""fudge_data.py -- Quetzal mirror of molgpt/molgpt_gfn/fudge_data_molgpt.py
and g2pt/g2pt_gfn/fudge_data_g2pt.py: generate labeled (hidden_state,
realized_atom_type, eventual_success) triples from unguided rollouts of the
frozen Quetzal prior, training data for a FUDGE-style future discriminator
baseline (see papers/current/decisions.md, "FUDGE baseline").

Reduced rollout budget relative to the two pilot legs (author instruction):
Quetzal's rollout is far more expensive per sample than MolGPT's/G2PT's --
autoregressive atom-type decisions AND a full diff_steps-step diffusion
process per atom's 3D coordinate, versus a single forward pass per token for
the pilots -- so this script defaults to a smaller --n than the pilots use.

No candidate-lookahead problem despite that per-sample cost, and this is the
whole reason the final FUDGE design (masked BCE on an h -> vocab_size
LogitGuide, not a per-candidate re-encoding) was chosen over the earlier,
abandoned per-candidate approach: Quetzal's own DB/RTB guide only ever needs
the hidden state `h` too (see gflow.py's LitGFlowNet -- `guided = prior_logits
+ guide(h)` for the bare LogitGuide architecture), never the state AFTER a
hypothetical next atom, which is what would have required re-running the
expensive diffusion coordinate sampler once per candidate atom type. This
script needs no diffusion sampling at all beyond what an ordinary
unconditional rollout already does.

Every decision here already IS an atom-type choice (or STOP) -- Quetzal's
guide has no separate atom-token-dimension mask the way MolGPT's/G2PT's do
(its entire vocabulary already is atom types by construction, see
guides.py's masked_guided_logits_dispatch docstring), so unlike the two
pilots, no additional per-step filtering to "atom decisions only" is needed
or applied here.

No guide, no gradient -- a single unguided rollout pass, built directly on
`LitGFlowNet` for its frozen-prior-loading and reward-function convenience,
but never touching its `guide`/`flow_head`/`logZ`/optimiser (irrelevant to
this script, which trains nothing and just collects data).

Usage: python fudge_data.py --reward osim --n 500 --out fudge_data/osim.pt
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from chem import Molecule, GEN, STOP, PAD, QM9_MASK  # noqa: E402
from gflow import LitGFlowNet, GFNConfig  # noqa: E402

# Mirrors scripts/01_train_guides.sh's reward_flags().
REWARD_FLAGS = {
    "osim": dict(reward="guacamol", reward_smiles="hard_osimertinib"),
    "peri": dict(reward="guacamol", reward_smiles="perindopril_rings"),
    "fexo": dict(reward="guacamol", reward_smiles="hard_fexofenadine"),
    "nitrogen": dict(reward="nitrogen_count"),
}


def _atom_mask(mask_atoms, device):
    if mask_atoms is None or mask_atoms == "None":
        mask = torch.ones(128, dtype=torch.bool, device=device)
        mask[GEN] = False
        mask[PAD] = False
    elif mask_atoms == "qm9":
        mask = QM9_MASK.to(device)
    else:
        raise ValueError(f"Unknown mask_atoms {mask_atoms!r}")
    return mask


@torch.no_grad()
def rollout_with_states(lit, bsz, max_len, device, mask_atoms):
    """Unguided rollout, mirroring gflow.py's `_generate_guided` with
    guide=None. Returns (per-trajectory stacked hidden states, per-trajectory
    realized atom-type ids, per-trajectory final log_reward)."""
    mask = _atom_mask(mask_atoms, device)
    atoms = torch.full((bsz, 1), GEN, dtype=torch.long, device=device)
    coords = torch.zeros(bsz, 1, 3, device=device)
    stop_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
    states = [[] for _ in range(bsz)]
    tokens = [[] for _ in range(bsz)]

    for _ in range(max_len):
        idx = torch.arange(atoms.shape[1], device=device).expand(bsz, -1)
        seq = lit.frozen.encode1(idx, atoms, coords)
        h = seq[:, -1, :]
        prior_logits = lit.frozen.proj_logits(h)
        masked = prior_logits.float().masked_fill(~mask, -1e9)
        behav = F.softmax(masked, dim=-1)
        next_atom = torch.multinomial(behav, num_samples=1)

        alive_idx = (~stop_mask).nonzero(as_tuple=True)[0].tolist()
        h_cpu, na_cpu = h.cpu(), next_atom.squeeze(-1).cpu()
        for i in alive_idx:
            states[i].append(h_cpu[i])
            tokens[i].append(na_cpu[i])

        stop_mask = stop_mask | (next_atom.squeeze(-1) == STOP)
        if stop_mask.all():
            break
        atoms = torch.cat([atoms, next_atom], dim=1)
        x = lit.frozen.encode2(atoms[:, 1:], seq)[:, -1, :]
        next_coord, _traj = lit.frozen.sample_coord(x, device=device, num_steps=lit.cfg.diff_steps)
        coords = torch.cat([coords, next_coord.view(bsz, 1, 3)], dim=1)

    mols = Molecule(atoms=atoms[:, 1:], coords=coords[:, 1:]).to("cpu").unbatch()
    log_reward = lit.compute_log_reward(mols).cpu().tolist()
    return states, tokens, log_reward


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward", required=True, choices=list(REWARD_FLAGS))
    ap.add_argument("--n", type=int, default=500,
                     help="reduced default vs. the pilots' 5000 -- see module docstring")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantile", type=float, default=0.75)
    ap.add_argument("--quetzal_ckpt", default="geom.ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = GFNConfig(name=f"fudge-data-{args.reward}", quetzal_ckpt=args.quetzal_ckpt,
                     objective="rtb", use_hidden_guide=False, **REWARD_FLAGS[args.reward])
    lit = LitGFlowNet(asdict(cfg)).to(args.device)
    lit.eval()

    all_states, all_tokens, all_rewards = [], [], []
    n_done = 0
    while n_done < args.n:
        bsz = min(args.chunk, args.n - n_done)
        states, tokens, log_reward = rollout_with_states(lit, bsz, cfg.max_len, args.device, cfg.mask_atoms)
        for s, t, r in zip(states, tokens, log_reward):
            if s:
                all_states.append(torch.stack(s))
                all_tokens.append(torch.stack(t))
                all_rewards.append(r)
        n_done += bsz
        print(f"[fudge_data] {n_done}/{args.n} rollouts done", flush=True)

    rewards_t = torch.tensor(all_rewards)
    # Same fix as the pilots' fudge_data scripts: quantile over VALID rewards
    # only, floor-scored trajectories always fail (see decisions.md / the
    # pilots' scripts for the failure mode this avoids).
    valid_mask = rewards_t > cfg.invalid_logr
    if valid_mask.sum() < 10:
        raise SystemExit(f"[FATAL] only {int(valid_mask.sum())} valid trajectories out of "
                          f"{len(all_rewards)} -- too few to set a meaningful success threshold")
    threshold = torch.quantile(rewards_t[valid_mask], args.quantile).item()
    labels = valid_mask & (rewards_t >= threshold)
    print(f"[fudge_data] reward={args.reward} valid={int(valid_mask.sum())}/{len(all_rewards)} "
          f"threshold(q={args.quantile} of valid)={threshold:.4f} "
          f"success_rate={labels.float().mean().item():.3f}")

    H = torch.cat(all_states, dim=0)
    tok = torch.cat(all_tokens, dim=0)
    y = torch.cat([labels[i].expand(all_states[i].shape[0]) for i in range(len(all_states))])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"h": H, "token": tok, "y": y, "threshold": threshold, "reward": args.reward,
                "quantile": args.quantile, "n_trajectories": len(all_states),
                "n_states": int(H.shape[0])}, args.out)
    print(f"[fudge_data] wrote {args.out}: {H.shape[0]} labeled decisions from {len(all_states)} trajectories")


if __name__ == "__main__":
    main()
