"""Aggregate every `final_dump_g2pt.py` summary.json and
`flip_g2pt.py` flip_report*.json under this leg's results/ into one
master_table.csv, mirroring `aggregate_dumps.py`'s column semantics where
they overlap (same discipline as `molgpt/molgpt_gfn/aggregate_molgpt.py`,
this leg's counterpart) -- one row per config
(reward, guide, objective, beta), metrics aggregated as mean/std across
training seeds, kept in a separate table rather than merged into the
Quetzal master_table.csv (different model, per the plan).

Run-name parsing matches `aggregate_dumps.py`'s `NAME_RE_SWEEP` exactly
(`sweep-<reward>-<guide>-<objective>-replay_<on|off>-b<beta>[-s<seed>]`),
since this leg's naming convention was deliberately copied from it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

NAME_RE = re.compile(
    r"^sweep-(?P<reward>[^-]+)-(?P<guide>[^-]+)-(?P<objective>[^-]+)-"
    r"replay_(?P<replay>on|off)-b(?P<beta>\d+)(?:-s(?P<seed>\d+))?$")


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps_root", default="results/g2pt/dumps")
    ap.add_argument("--flips_root", default="results/g2pt/flips")
    ap.add_argument("--flip_temp_key", default="t1.0")
    ap.add_argument("--flip_regime", default="atom_steps",
                     help="Which of flip_g2pt.py's by_regime keys (overall/"
                          "atom_steps/non_atom_steps) to report as the primary "
                          "*_mean/_std columns. Default atom_steps: the regime "
                          "where the guide is actually active under Option A's "
                          "step-gating, so it's the fair comparison point "
                          "against MolGPT's flip metrics (MolGPT has no "
                          "step-gating -- masking is dimension-wise at every "
                          "position -- so its flip report has no by_regime "
                          "split at all, its one number already is this).")
    ap.add_argument("--out_dir", default="results/g2pt/_aggregate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- collect dump summaries, one per (base_name, seed) ----
    by_base = defaultdict(list)   # base_name -> list of (seed, summary_dict)
    for f in sorted(glob.glob(f"{args.dumps_root}/*/summary.json")):
        summary = json.load(open(f))
        name = summary["name"]
        m = NAME_RE.match(name)
        if not m:
            print(f"[skip] {name!r} doesn't match the sweep-* naming convention")
            continue
        base_name = name if m.group("seed") is None else name[:name.rfind("-s")]
        by_base[base_name].append((m, summary))

    # ---- collect flip reports, keyed the same way ----
    flips_by_base = defaultdict(list)
    for f in sorted(glob.glob(f"{args.flips_root}/flip_report_*.json")):
        report = json.load(open(f))
        name = report["label"]
        m = NAME_RE.match(name)
        if not m:
            continue
        base_name = name if m.group("seed") is None else name[:name.rfind("-s")]
        t = report.get("results_by_temp", {}).get(args.flip_temp_key)
        # Fixed 2026-09-04: this used to append `t` directly and read
        # `t.get("delivered_frac")` below -- silently always None, since
        # flip_g2pt.py's own report nests every metric one level deeper than
        # flip_molgpt.py's, under `t["by_regime"][<regime>]` (overall/
        # atom_steps/non_atom_steps), not at `t` itself. Every master_table.csv
        # built before this fix has empty delivered_frac/argmax_flip_rate/etc
        # columns -- re-run aggregate after this fix to backfill them.
        by_regime = (t or {}).get("by_regime", {})
        if args.flip_regime in by_regime:
            flips_by_base[base_name].append(by_regime[args.flip_regime])

    rows = []
    for base_name, entries in sorted(by_base.items()):
        m0 = entries[0][0]
        row = {
            "name": base_name, "reward": m0.group("reward"), "guide": m0.group("guide"),
            "objective": m0.group("objective"), "replay": m0.group("replay"),
            "beta": int(m0.group("beta")), "n_seeds": len(entries),
        }
        for source in ("base", "guided"):
            for metric in ("parse_rate", "uniqueness", "log_reward_mean",
                           "log_reward_top1", "log_reward_top10", "log_reward_top100"):
                vals = [s.get(source, {}).get(metric) for _m, s in entries]
                mean, std = _mean_std(vals)
                row[f"{source}_{metric}_mean"] = mean
                row[f"{source}_{metric}_std"] = std
        for metric in ("log_reward_mean", "log_reward_top1", "log_reward_top10", "log_reward_top100"):
            deltas = []
            for _m, s in entries:
                b, g = s.get("base", {}).get(metric), s.get("guided", {}).get(metric)
                if b is not None and g is not None:
                    deltas.append(g - b)
            mean, std = _mean_std(deltas)
            row[f"{metric}_delta_mean"] = mean
            row[f"{metric}_delta_std"] = std

        flip_entries = flips_by_base.get(base_name, [])
        for metric in ("delivered_frac", "argmax_flip_rate", "sample_flip_rate",
                       "mean_total_variation", "mean_KL", "mean_prior_top1_gap"):
            vals = [f.get(metric) for f in flip_entries]
            mean, std = _mean_std(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        row["n_flip_seeds"] = len(flip_entries)

        rows.append(row)

    if not rows:
        print("[warn] no matching dump summaries found -- nothing to aggregate")
        return

    fieldnames = list(rows[0].keys())
    with open(out_dir / "master_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_dir / 'master_table.csv'} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
