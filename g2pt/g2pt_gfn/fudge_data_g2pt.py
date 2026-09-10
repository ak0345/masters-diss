"""fudge_data_g2pt.py -- generate labeled (hidden_state, realized_token,
eventual_success) triples from unguided rollouts of the frozen G2PT prior:
training data for a FUDGE-style future discriminator baseline (see
papers/current/decisions.md, "FUDGE baseline", and
molgpt/molgpt_gfn/fudge_data_molgpt.py's docstring for the full rationale --
this is the G2PT mirror of that script, using G2PT's own clean atom-decision
step gate instead of a realized-token check).

Only atom-decision steps are kept (`prior.is_atom_step`, true iff the
previous token was `<boc>`/`<sepc>`) -- G2PT's own DB/RTB guides are gated
identically (`atom_guide_only`), so the discriminator this trains is never
read anywhere else, and every non-atom step's example would supervise a
dimension nothing ever looks at.

No guide, no gradient -- a single unguided rollout pass, reimplemented
standalone rather than via Trainer (see the MolGPT version's docstring for
why: Trainer's __init__ unconditionally builds a guide/flow-head/optimiser
this script has no use for).

Usage: python fudge_data_g2pt.py --reward osim --n 5000 --out results/g2pt/fudge_data/osim.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GFNConfig  # noqa: E402
from g2pt_prior import FrozenG2PT, G2PTConfig, seq_to_smiles  # noqa: E402
from reward_adapter import build_reward_smiles  # noqa: E402
from gflow_g2pt import _decode  # noqa: E402

# Mirrors scripts_g2pt/01_train_guides.sh's reward_flags().
REWARD_FLAGS = {
    "osim": dict(reward="guacamol", reward_smiles="hard_osimertinib"),
    "peri": dict(reward="guacamol", reward_smiles="perindopril_rings"),
    "fexo": dict(reward="guacamol", reward_smiles="hard_fexofenadine"),
    "nitrogen": dict(reward="nitrogen_count"),
}


@torch.no_grad()
def rollout_with_states(prior, reward_fn, bsz, max_len, device):
    """Unguided rollout. Returns (per-trajectory stacked hidden states,
    per-trajectory realized atom-token ids, per-trajectory final log_reward)
    -- kept ONLY at atom-decision steps."""
    x = torch.full((bsz, 1), prior.boc_id, dtype=torch.long, device=device)
    alive = torch.ones(bsz, dtype=torch.bool, device=device)
    states = [[] for _ in range(bsz)]
    tokens = [[] for _ in range(bsz)]
    for _t in range(max_len - 1):
        prev_token = x[:, -1]
        is_atom = prior.is_atom_step(prev_token)
        h = prior.hidden_states(x)[:, -1, :]
        prior_logits = prior.mask_logits(prior.proj_logits(h))
        probs = F.softmax(prior_logits, dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(-1)
        keep = (alive & is_atom).nonzero(as_tuple=True)[0].tolist()
        h_cpu, nxt_cpu = h.cpu(), nxt.cpu()
        for i in keep:
            states[i].append(h_cpu[i])
            tokens[i].append(nxt_cpu[i])
        alive = alive & (nxt != prior.eog_id)
        x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
        if not alive.any():
            break
    # _decode returns G2PT's own grammar token string, not a SMILES string --
    # must go through seq_to_smiles first, matching Trainer.compute_log_reward's
    # own convention (see decisions.md's "wrong reward input" fix).
    seqs = [_decode(prior, row) for row in x[:, 1:].tolist()]
    smiles = [seq_to_smiles(s) or "" for s in seqs]
    log_reward = [reward_fn(s) for s in smiles]
    return states, tokens, log_reward


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward", required=True, choices=list(REWARD_FLAGS))
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantile", type=float, default=0.75)
    ap.add_argument("--model_name_or_path", default="checkpoints/g2pt_geom_drugs_hf")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = GFNConfig(model_name_or_path=args.model_name_or_path, **REWARD_FLAGS[args.reward])
    prior = FrozenG2PT(G2PTConfig(model_name_or_path=cfg.model_name_or_path), device=args.device)
    prior.eval()
    reward_fn = build_reward_smiles(cfg)

    all_states, all_tokens, all_rewards = [], [], []
    n_done = 0
    while n_done < args.n:
        bsz = min(args.chunk, args.n - n_done)
        states, tokens, log_reward = rollout_with_states(prior, reward_fn, bsz, cfg.max_len, args.device)
        for s, t, r in zip(states, tokens, log_reward):
            if s:
                all_states.append(torch.stack(s))
                all_tokens.append(torch.stack(t))
                all_rewards.append(r)
        n_done += bsz
        print(f"[fudge_data] {n_done}/{args.n} rollouts done", flush=True)

    rewards_t = torch.tensor(all_rewards)
    # Quantile computed over VALID rewards only, and floor-scored ("invalid")
    # trajectories always fail regardless of where the quantile lands. Caught
    # via a smoke test: with ~25% validity, ~75% of rewards tie exactly at
    # cfg.invalid_logr, so an unconditional quantile(0.75) over ALL rewards
    # can land ON the floor -- ">= threshold" then includes nearly everyone
    # (93% success in the failing run) instead of the intended top quartile.
    # Same risk exists for MolGPT whenever its own validity dips below ~75%,
    # so this fix is mirrored in fudge_data_molgpt.py too, not just here.
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
    # Continuous target for the regression variant (train_fudge_g2pt.py --target r):
    # each state's realized trajectory's own log-reward, clipped at the invalid floor.
    # Kept alongside the binary `y` rather than replacing it, so old (classification)
    # runs stay reproducible from the same file.
    r_traj = rewards_t.clamp(min=cfg.invalid_logr)
    r = torch.cat([r_traj[i].expand(all_states[i].shape[0]) for i in range(len(all_states))])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"h": H, "token": tok, "y": y, "r": r, "threshold": threshold,
                "reward": args.reward, "quantile": args.quantile,
                "n_trajectories": len(all_states), "n_states": int(H.shape[0])}, args.out)
    print(f"[fudge_data] wrote {args.out}: {H.shape[0]} labeled atom-decisions from {len(all_states)} trajectories")


if __name__ == "__main__":
    main()
