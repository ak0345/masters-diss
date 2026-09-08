"""Final evaluation dump for a trained G2PT guide -- see
`molgpt/molgpt_gfn/final_dump_molgpt.py`'s module docstring for the full
convention audit against `final_dump.py` (N=5000, `guide_source=ema`
default, `sample_temp=cfg.sample_temp` not a clean temp=1 eval,
`rand_eps=0.0`, mean-of-top-k reward columns, both `guided` and `base`
dumped at the same N). Same convention here, adapted only where G2PT's
output shape differs from MolGPT's:

  - `generate_guided`'s `"seqs"` are space-joined graph-token strings, not
    SMILES directly -- each one needs `seq_to_smiles` (parses `<boc>`..`<eoc>`
    node declarations and `<bog>`..`<eog>` bond declarations into an RDKit
    Mol, then canonicalises) before it can be scored as a SMILES string or
    counted for uniqueness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GFNConfig  # noqa: E402
from gflow_g2pt import Trainer  # noqa: E402
from g2pt_prior import seq_to_smiles_classified  # noqa: E402


def load_trained_guide(trainer: Trainer, ckpt_path: str, guide_source: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if guide_source == "ema":
        sd_key = "guide_ema_state_dict"
    elif guide_source == "policy":
        sd_key = "guide_state_dict"
    else:
        raise ValueError(f"guide_source must be 'ema' or 'policy', got {guide_source!r}")
    if sd_key not in ckpt:
        raise SystemExit(f"[FATAL] {ckpt_path} has no {sd_key!r} -- retrain with EMA support.")
    missing, _unexpected = trainer.guide.load_state_dict(ckpt[sd_key], strict=False)
    if missing:
        raise SystemExit(
            f"[FATAL] {len(missing)} guide weights did not load from {ckpt_path} "
            f"(e.g. {missing[:4]}). Refusing to dump an untrained guide.")


@torch.no_grad()
def dump(trainer: Trainer, name: str, guide, n: int, sample_temp: float,
         rand_eps: float, chunk: int, out_dir: Path) -> dict:
    all_smiles, all_class, all_logr = [], [], []
    remaining = n
    while remaining > 0:
        b = min(chunk, remaining)
        out = trainer.generate_guided(b, guide, sample_temp=sample_temp, rand_eps=rand_eps)
        for seq in out["seqs"]:
            smi, reason = seq_to_smiles_classified(seq)
            all_smiles.append(smi)
            all_class.append(reason)
        all_logr.extend(out["log_reward"].tolist())
        remaining -= b

    valid_mask = [c == "valid" for c in all_class]
    valid_smiles = [s for s, v in zip(all_smiles, valid_mask) if v]
    logr = np.array(all_logr, dtype=np.float64)
    valid_logr = logr[np.array(valid_mask, dtype=bool)] if len(logr) else np.array([])
    n_syntax_fail = sum(1 for c in all_class if c == "syntax_fail")
    n_valence_fail = sum(1 for c in all_class if c == "valence_fail")

    parse_rate = len(valid_smiles) / max(n, 1)
    stats = {
        "n_generated": n,
        "n_valid_smiles": len(valid_smiles),
        "parse_rate": parse_rate,
        # author instruction, 2026-09-04: syntax vs valence failures tracked
        # separately -- "syntax_fail" here means _seq_to_mol's own grammar
        # check failed (missing <boc>/<eoc>/<bog>/<eog>, malformed sections),
        # before RDKit sanitization was even attempted -- see
        # g2pt_prior.seq_to_smiles_classified's docstring.
        "n_syntax_fail": n_syntax_fail,
        "n_valence_fail": n_valence_fail,
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
          f"syntax_fail={n_syntax_fail} valence_fail={n_valence_fail} "
          f"logR_mean={stats['log_reward_mean']} top10={stats['log_reward_top10']}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--guide_source", choices=["ema", "policy"], default="ema")
    ap.add_argument("--sample_temp", type=float, default=None)
    ap.add_argument("--rand_eps", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GFNConfig(**ckpt["config"])
    trainer = Trainer(cfg, device=args.device)
    load_trained_guide(trainer, args.ckpt, args.guide_source)
    trainer.guide.eval()

    sample_temp = args.sample_temp if args.sample_temp is not None else cfg.sample_temp
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] reward={cfg.reward} reward_smiles={cfg.reward_smiles} "
          f"sample_temp={sample_temp} guide_source={args.guide_source} n={args.n} "
          f"max_len={cfg.max_len}")

    summary = {
        "ckpt": args.ckpt, "name": Path(args.ckpt).parent.parent.name,
        "seed": args.seed, "n_requested": args.n, "guide_source": args.guide_source,
        "reward": cfg.reward, "reward_smiles": cfg.reward_smiles, "sample_temp": sample_temp,
        "guided": dump(trainer, "guided", trainer.guide, args.n, sample_temp, args.rand_eps, args.chunk, out_dir),
    }
    if not args.skip_base:
        summary["base"] = dump(trainer, "base", None, args.n, sample_temp, args.rand_eps, args.chunk, out_dir)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
