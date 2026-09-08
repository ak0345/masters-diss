"""make_comparison_table.py -- one CSV merging the key summary numbers from
all three legs, one row per (model, reward), for direct use in the
dissertation's comparison table/prose.

Columns: model, reward, n_configs, guided/base parse_rate, guided/base
uniqueness, guided top-10 score (raw [0,1] scale -- see make_cmp01_landscape's
docstring for how MolGPT/G2PT's log-reward is converted), delivered_frac,
argmax_flip_rate, sample_flip_rate (G2PT's `atom_steps` regime, others'
only regime), plus the shared GEOM-Drugs best-of-10k and published
(Brown et al. 2019) reference scores for that reward, repeated on every row
for convenient side-by-side reading.
"""
import csv
import math
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (rel, load_master, flip_reports, flip_metric,
                               json, COMMON_REWARDS, PUBLISHED, GEOM_BASELINE_PATH,
                               NAME_RE)

REWARD_NORM = {"osimertinib": "osim", "perindopril": "peri"}


def quetzal_row_ok(row):
    if not row["name"].startswith("sweep-"):
        return False
    return row.get("guide") in ("base", "hidden") and row.get("objective") in ("db", "rtb") \
        and row.get("replay") == "off" and row.get("beta") == "10"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    geom_baseline = json.load(open(GEOM_BASELINE_PATH))
    out_rows = []

    for model in ("Quetzal", "MolGPT", "G2PT"):
        master = load_master(model)
        reports = flip_reports(model)

        for reward in COMMON_REWARDS:
            if model == "Quetzal":
                rows = [r for r in master if quetzal_row_ok(r)
                        and REWARD_NORM.get(r.get("reward"), r.get("reward")) == reward]
            else:
                rows = [r for r in master if r.get("reward") == reward]
            if not rows:
                continue

            if model == "Quetzal":
                top10_scores = [float(r["guided_reward_top10_mean"]) for r in rows
                                 if r.get("guided_reward_top10_mean")]
            else:
                top10_scores = [math.exp(float(r["guided_log_reward_top10_mean"]))
                                 for r in rows if r.get("guided_log_reward_top10_mean")]

            def col(source, metric):
                key = f"{source}_{metric}_mean"
                return mean([float(r[key]) for r in rows if r.get(key) not in (None, "", "None")])

            # MolGPT's/G2PT's master_table.csv rows are seed-collapsed base
            # names (e.g. "sweep-fexo-hidden-db-replay_off-b10"); Quetzal's
            # own aggregate keeps one row PER SEED instead (confirmed by
            # reading it directly -- `name` already includes "-s0"/"-s42",
            # `base_name` is the separate collapsed column this script
            # doesn't need). Flip report labels always carry the seed
            # suffix, so: strip-and-match for the two pilots, exact-match
            # for Quetzal.
            if model == "Quetzal":
                matching_names = {r["name"] for r in rows}
            else:
                matching_names = {r["name"] for r in rows}  # already base names
            flip_vals = {m: [] for m in ("delivered_frac", "argmax_flip_rate", "sample_flip_rate")}
            for label, report in reports.items():
                key = label if model == "Quetzal" else re.sub(r"-s\d+$", "", label)
                if key not in matching_names:
                    continue
                for metric in flip_vals:
                    v = flip_metric(report, model, metric)
                    if isinstance(v, (int, float)):
                        flip_vals[metric].append(v)

            gb = geom_baseline.get(reward, {}).get("top10")
            pub = PUBLISHED.get(reward, {})

            out_rows.append({
                "model": model, "reward": reward, "n_configs": len(rows),
                "base_parse_rate": col("base", "parse_rate"),
                "guided_parse_rate": col("guided", "parse_rate"),
                "base_uniqueness": col("base", "uniqueness"),
                "guided_uniqueness": col("guided", "uniqueness"),
                "guided_top10_score": mean(top10_scores),
                "delivered_frac": mean(flip_vals["delivered_frac"]),
                "argmax_flip_rate": mean(flip_vals["argmax_flip_rate"]),
                "sample_flip_rate": mean(flip_vals["sample_flip_rate"]),
                "geom_best_of_10k_top10": gb,
                "published_reinvent_smiles": pub.get("REINVENT SMILES"),
                "published_chembl_best_of_dataset": pub.get("ChEMBL best-of-dataset"),
            })

    out_path = rel("comparison_figures", "comparison_table.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
