"""fudge_data_molgpt.py -- generate labeled (hidden_state, eventual_success)
pairs from unguided rollouts of the frozen MolGPT prior: training data for a
FUDGE-style future discriminator baseline (see papers/current/decisions.md,
"FUDGE baseline" -- an alternative logit-reweighting scheme trained by
ordinary binary cross-entropy on partial-state -> eventual-outcome pairs,
rather than a GFlowNet objective like the DB/RTB guides).

No guide, no gradient -- a single unguided rollout pass. Reimplemented
standalone here (not calling Trainer.generate_guided(guide=None, ...))
because Trainer's __init__ unconditionally builds a guide, a flow head or
logZ, and an optimiser, none of which this script needs; instantiating one
just to throw it away would be misleading about what this script does.

Label: whether a trajectory's final reward lands at or above a quantile of
this same unguided sample (default: top quartile). Adaptive per model and
reward rather than a fixed absolute cutoff, since achievable reward ranges
differ across models and rewards -- see cmp07_bestofn_baseline.py's base
pools for exactly this same range variation. Every per-step hidden state
along a trajectory inherits that trajectory's own eventual label: this is
FUDGE's "future discriminator" idea directly -- a lone partial state may
look ambiguous, but it is a perfectly valid training example for what it
eventually turned into.

Also records which token was actually realized at each collected state
(only states where that token is an atom token are kept at all -- see
train_fudge_molgpt.py's docstring for why: the discriminator is only ever
read at atom-vocab dimensions at inference, so a non-atom-realized state
supervises a dimension nothing ever looks at). This is what lets the
discriminator be trained as a single `h -> vocab_size` LogitGuide (masked
binary cross-entropy on just the realized dimension) instead of needing a
separate forward pass per hypothetical candidate token.

Usage: python fudge_data_molgpt.py --reward osim --n 5000 --out results/molgpt/fudge_data/osim.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GFNConfig  # noqa: E402
from molgpt_prior import FrozenMolGPT, MolGPTConfig  # noqa: E402
from reward_adapter import build_reward_smiles  # noqa: E402
from gflow_molgpt import _decode  # noqa: E402

# Mirrors scripts_molgpt/01_train_guides.sh's reward_flags() -- same reward
# names, same underlying GuacaMol benchmark keys, so a FUDGE run on "osim"
# is scored against exactly the same objective the DB/RTB sweep used.
REWARD_FLAGS = {
    "osim": dict(reward="guacamol", reward_smiles="hard_osimertinib"),
    "peri": dict(reward="guacamol", reward_smiles="perindopril_rings"),
    "fexo": dict(reward="guacamol", reward_smiles="hard_fexofenadine"),
    "nitrogen": dict(reward="nitrogen_count"),
}


@torch.no_grad()
def rollout_with_states(prior, reward_fn, bsz, max_len, device):
    """Unguided rollout. Returns (per-trajectory stacked hidden states,
    per-trajectory chosen-token ids, per-trajectory final log_reward) --
    states/tokens kept ONLY where the realized next token is an atom token
    (see module docstring). A trajectory's lists have exactly as many rows
    as atom decisions it made."""
    x = torch.full((bsz, 1), prior.seed_idx, dtype=torch.long, device=device)
    alive = torch.ones(bsz, dtype=torch.bool, device=device)
    states = [[] for _ in range(bsz)]
    tokens = [[] for _ in range(bsz)]
    for _t in range(max_len - 1):
        h = prior.hidden_states(x)[:, -1, :]
        prior_logits = prior.proj_logits(h)
        probs = F.softmax(prior_logits, dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(-1)
        is_atom = prior.atom_token_mask[nxt]
        keep = (alive & is_atom).nonzero(as_tuple=True)[0].tolist()
        h_cpu, nxt_cpu = h.cpu(), nxt.cpu()
        for i in keep:
            states[i].append(h_cpu[i])
            tokens[i].append(nxt_cpu[i])
        alive = alive & (nxt != prior.pad_idx)
        x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
        if not alive.any():
            break
    smiles = [_decode(prior, row) for row in x[:, 1:].tolist()]
    log_reward = [reward_fn(s) for s in smiles]
    return states, tokens, log_reward


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward", required=True, choices=list(REWARD_FLAGS))
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantile", type=float, default=0.75,
                     help="reward quantile at/above which a trajectory is labelled success")
    ap.add_argument("--molgpt_ckpt", default="checkpoints/molgpt_geom_drugs_unconditional.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = GFNConfig(molgpt_ckpt=args.molgpt_ckpt, **REWARD_FLAGS[args.reward])
    molcfg = MolGPTConfig(vocab_size=cfg.vocab_size, block_size=cfg.max_len)
    prior = FrozenMolGPT(molcfg, cfg.molgpt_ckpt, device=args.device)
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
    # trajectories always fail regardless of where the quantile lands -- see
    # fudge_data_g2pt.py's identical fix for the failure mode this avoids
    # (an unconditional quantile can land exactly on the invalid floor
    # whenever validity is below ~(1 - quantile), making ">= threshold"
    # include nearly everyone instead of the intended top quartile).
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
    print(f"[fudge_data] wrote {args.out}: {H.shape[0]} labeled atom-decisions from {len(all_states)} trajectories")


if __name__ == "__main__":
    main()
