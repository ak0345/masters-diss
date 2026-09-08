"""Final evaluation dump for a trained MolGPT guide -- the `final_dump.py`
equivalent for this leg, matching its convention exactly rather than
inventing a different one (audited against `final_dump.py` directly,
2026-09-04, see `papers/current/decisions.md` and `../vendor/PROVENANCE.md`):

  - N=5000 molecules (`--n`, default matches `final_dump.py`'s own default).
  - `--guide_source` defaults to `"ema"` -- every published Quetzal number
    this pilot is compared against uses the EMA guide, not the raw trained
    one (`final_dump.py --guide_source` also defaults to `"ema"`).
  - **`sample_temp` defaults to the run's own `cfg.sample_temp`** (2.0, the
    training exploration temperature), NOT a "clean" temp=1 deployment
    setting -- this is what `final_dump.py` actually does
    (`sample_temp=lit.cfg.sample_temp` unless `--sample_temp` overrides it),
    confirmed by reading the source rather than assumed. `--rand_eps`
    defaults to 0.0, matching `final_dump.py`.
  - Both `guided` (the loaded guide) and `base` (the frozen prior, no guide)
    are dumped at the same N, so terminal-reward deltas are computed against
    a same-N baseline, not a differently-sized one.
  - Metrics: `n_generated, n_valid_smiles, parse_rate, uniqueness,
    log_reward_mean, log_reward_top1, log_reward_top10, log_reward_top100`
    (top10/top100 are the MEAN of the top 10/100 values, matching
    `final_dump.py`'s `np.mean(np.sort(x)[-10:])` convention exactly, not
    "the 10th-best value").

**Deliberately not ported**: `atom_stability`/`mol_stability` (EDM 3D-geometry
stability metrics -- there is no conformer here to be stable or not, the
concept doesn't apply to a SMILES-token model) and FCD/Wasserstein-vs-reference
(optional in `final_dump.py` itself, dropped here for scope -- worth adding
later against a GuacaMol-appropriate reference set, but GEOM-Drugs, Quetzal's
own reference, would be the wrong comparison distribution for a
GuacaMol-pretrained model).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GFNConfig  # noqa: E402
from gflow_molgpt import Trainer, _decode  # noqa: E402
from reward_adapter import classify_smiles  # noqa: E402


def load_trained_guide(trainer: Trainer, ckpt_path: str, guide_source: str) -> None:
    """Overwrite `trainer.guide`'s (freshly zero-initialised) weights with
    the real trained ones from `ckpt_path`, in place. Same fail-loudly
    discipline as flip_molgpt.py's loader."""
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
        for smi in out["smiles"]:
            # already decoded SMILES strings (MolGPT's own generate_guided
            # decodes at PAD-truncation time via _decode, see gflow_molgpt.py)
            all_smiles.append(smi)
            all_class.append(classify_smiles(smi))
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
        # separately -- see reward_adapter.classify_smiles's docstring.
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
    ap.add_argument("--sample_temp", type=float, default=None,
                     help="default: the run's own cfg.sample_temp -- see module docstring")
    ap.add_argument("--rand_eps", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=256)
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
          f"sample_temp={sample_temp} guide_source={args.guide_source} n={args.n}")

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
