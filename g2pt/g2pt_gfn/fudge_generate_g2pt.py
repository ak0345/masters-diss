"""fudge_generate_g2pt.py -- generate + score molecules under a trained
FUDGE discriminator (train_fudge_g2pt.py's output), writing the exact output
shape final_dump_g2pt.py uses (summary.json, {name}_smiles.txt,
{name}_rewards.npy) so every existing comparison_figures/ and per-leg figure
script can read a FUDGE run exactly like a DB/RTB config. See
molgpt/molgpt_gfn/fudge_generate_molgpt.py's docstring for why this
reimplements the rollout loop rather than retrofitting
masked_guided_logits_dispatch.

G2PT-specific: the candidate lookahead only runs at atom-decision steps
(`prior.is_atom_step`), matching the DB/RTB guides' own `atom_guide_only`
step-level gate -- at every other step, guided logits equal the prior's
exactly, same discipline as gflow_g2pt.py's `_guided_step`. Candidates
within an atom-decision step are further restricted to atom-token ids
(`atom_token_mask`), matching the dimension-wise mask too. `mask_logits` is
applied throughout (G2PT's padded vocab needs it; MolGPT's doesn't).

Usage: python fudge_generate_g2pt.py --disc logs/g2pt-fudge/osim/discriminator.pt \
         --strength 1.0 --n 5000 --out_dir results/g2pt/fudge_dumps/osim
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GFNConfig  # noqa: E402
from g2pt_prior import FrozenG2PT, G2PTConfig, seq_to_smiles_classified  # noqa: E402
from reward_adapter import build_reward_smiles  # noqa: E402
from gflow_g2pt import _decode  # noqa: E402
from guides import LogFlowHead  # noqa: E402
from fudge_data_g2pt import REWARD_FLAGS  # noqa: E402


@torch.no_grad()
def fudge_rollout(prior, discriminator, reward_fn, bsz, max_len, device,
                   strength, sample_temp, rand_eps):
    atom_ids = prior.atom_token_mask.nonzero(as_tuple=True)[0]
    A = atom_ids.numel()
    x = torch.full((bsz, 1), prior.boc_id, dtype=torch.long, device=device)
    alive = torch.ones(bsz, dtype=torch.bool, device=device)
    for _t in range(max_len - 1):
        prev_token = x[:, -1]
        is_atom = prior.is_atom_step(prev_token)
        B, T = x.shape
        h = prior.hidden_states(x)[:, -1, :]
        prior_logits = prior.mask_logits(prior.proj_logits(h))

        guided_logits = prior_logits.clone()
        atom_rows = (alive & is_atom).nonzero(as_tuple=True)[0]
        if atom_rows.numel() > 0:
            Ba = atom_rows.numel()
            x_sel = x[atom_rows]
            x_rep = x_sel.unsqueeze(1).expand(Ba, A, T).reshape(Ba * A, T)
            cand = atom_ids.view(1, A).expand(Ba, A).reshape(-1)
            x_ext = torch.cat([x_rep, cand.unsqueeze(1)], dim=1)
            h_ext = prior.hidden_states(x_ext)[:, -1, :]
            disc_logit = discriminator(h_ext).view(Ba, A)
            guided_logits[atom_rows.unsqueeze(1), atom_ids.unsqueeze(0)] += strength * disc_logit

        behav = F.softmax(guided_logits / sample_temp, dim=-1)
        if rand_eps > 0:
            behav = (1 - rand_eps) * behav + rand_eps * prior.real_token_mask.float() / prior.real_token_mask.sum()
        nxt = torch.multinomial(behav, 1).squeeze(-1)
        alive = alive & (nxt != prior.eog_id)
        x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
        if not alive.any():
            break

    seqs = [_decode(prior, row) for row in x[:, 1:].tolist()]
    results = [seq_to_smiles_classified(s) for s in seqs]
    smi_list = [smi or "" for smi, _ in results]
    class_list = [reason for _, reason in results]
    log_reward = [reward_fn(s) for s in smi_list]
    return smi_list, class_list, log_reward


@torch.no_grad()
def dump_unguided(prior, reward_fn, n, max_len, device, chunk):
    all_smiles, all_class, all_logr = [], [], []
    remaining = n
    while remaining > 0:
        b = min(chunk, remaining)
        x = torch.full((b, 1), prior.boc_id, dtype=torch.long, device=device)
        alive = torch.ones(b, dtype=torch.bool, device=device)
        for _t in range(max_len - 1):
            h = prior.hidden_states(x)[:, -1, :]
            probs = F.softmax(prior.mask_logits(prior.proj_logits(h)), dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
            alive = alive & (nxt != prior.eog_id)
            x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
            if not alive.any():
                break
        seqs = [_decode(prior, row) for row in x[:, 1:].tolist()]
        results = [seq_to_smiles_classified(s) for s in seqs]
        for smi, reason in results:
            all_smiles.append(smi or "")
            all_class.append(reason)
        all_logr.extend(reward_fn(smi or "") for smi, _ in results)
        remaining -= b
    return all_smiles, all_class, all_logr


def stats_and_write(name, all_smiles, all_class, all_logr, n, out_dir):
    valid_mask = [c == "valid" for c in all_class]
    valid_smiles = [s for s, v in zip(all_smiles, valid_mask) if v]
    logr = np.array(all_logr, dtype=np.float64)
    valid_logr = logr[np.array(valid_mask, dtype=bool)] if len(logr) else np.array([])
    n_syntax_fail = sum(1 for c in all_class if c == "syntax_fail")
    n_valence_fail = sum(1 for c in all_class if c == "valence_fail")
    parse_rate = len(valid_smiles) / max(n, 1)
    stats = {
        "n_generated": n, "n_valid_smiles": len(valid_smiles), "parse_rate": parse_rate,
        "n_syntax_fail": n_syntax_fail, "n_valence_fail": n_valence_fail,
        "syntax_fail_rate": n_syntax_fail / max(n, 1),
        "valence_fail_rate": n_valence_fail / max(n, 1),
        "uniqueness": (len(set(valid_smiles)) / len(valid_smiles)) if valid_smiles else 0.0,
        "log_reward_mean": float(valid_logr.mean()) if len(valid_logr) else None,
        "log_reward_top1": float(valid_logr.max()) if len(valid_logr) else None,
        "log_reward_top10": float(np.mean(np.sort(valid_logr)[-10:])) if len(valid_logr) >= 10 else None,
        "log_reward_top100": float(np.mean(np.sort(valid_logr)[-100:])) if len(valid_logr) >= 100 else None,
    }
    with open(out_dir / f"{name}_smiles.txt", "w") as f:
        f.write("\n".join(valid_smiles) + ("\n" if valid_smiles else ""))
    np.save(out_dir / f"{name}_rewards.npy", valid_logr)
    print(f"[{name}] valid={len(valid_smiles)}/{n} ({parse_rate:.3f}) "
          f"logR_mean={stats['log_reward_mean']} top10={stats['log_reward_top10']}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--disc", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--sample_temp", type=float, default=1.0,
                     help="G2PT's own leg default (see decisions.md -- eval "
                          "temp fixed to 1.0 for this leg, not the shared 2.0)")
    ap.add_argument("--rand_eps", type=float, default=0.0)
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--model_name_or_path", default="checkpoints/g2pt_geom_drugs_hf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    disc_ckpt = torch.load(args.disc, map_location="cpu")
    reward = disc_ckpt["reward"]
    cfg = GFNConfig(model_name_or_path=args.model_name_or_path, **REWARD_FLAGS[reward])
    prior = FrozenG2PT(G2PTConfig(model_name_or_path=cfg.model_name_or_path), device=args.device)
    prior.eval()
    reward_fn = build_reward_smiles(cfg)

    discriminator = LogFlowHead(disc_ckpt["d_model"], disc_ckpt["hidden"], disc_ckpt["layers"]).to(args.device)
    discriminator.load_state_dict(disc_ckpt["state_dict"])
    discriminator.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] reward={reward} strength={args.strength} sample_temp={args.sample_temp} "
          f"n={args.n} disc_val_acc={disc_ckpt.get('val_acc')}")

    all_smiles, all_class, all_logr = [], [], []
    remaining = args.n
    while remaining > 0:
        b = min(args.chunk, remaining)
        smi_list, class_list, log_reward = fudge_rollout(
            prior, discriminator, reward_fn, b, cfg.max_len, args.device,
            args.strength, args.sample_temp, args.rand_eps)
        all_smiles.extend(smi_list)
        all_class.extend(class_list)
        all_logr.extend(log_reward)
        remaining -= b
        print(f"[fudge_generate] {args.n - remaining}/{args.n}", flush=True)

    summary = {
        "disc_ckpt": args.disc, "reward": reward, "strength": args.strength,
        "sample_temp": args.sample_temp, "n_requested": args.n, "seed": args.seed,
        "guided": stats_and_write("guided", all_smiles, all_class, all_logr, args.n, out_dir),
    }
    if not args.skip_base:
        b_smiles, b_class, b_logr = dump_unguided(prior, reward_fn, args.n, cfg.max_len, args.device, args.chunk)
        summary["base"] = stats_and_write("base", b_smiles, b_class, b_logr, args.n, out_dir)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
