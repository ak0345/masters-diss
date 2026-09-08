"""make_cmp04_validity.py -- three-way comparison: parse rate and uniqueness
of guided generation, base (frozen prior) vs guided, by model.

Reads each model's own master_table.csv (already seed-aggregated), restricted
to the same pilot-comparable grid as the other comparison figures (Quetzal:
sweep-*, guide in {base,hidden}, objective in {db,rtb}, replay=off, beta=10;
MolGPT/G2PT: their whole grid, already this scope), averaged again across
every remaining config so one bar per (model, base/guided) pair.
"""
import csv
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (rel, load_master, savefig, require, plt,
                               MODEL_COLOURS, NAME_RE, MASTER_TABLES)

REWARD_NORM = {"osimertinib": "osim", "perindopril": "peri"}


def quetzal_ok(row):
    if not row["name"].startswith("sweep-"):
        return False
    return row.get("guide") in ("base", "hidden") and row.get("objective") in ("db", "rtb") \
        and row.get("replay") == "off" and row.get("beta") == "10"


def quetzal_base_stats():
    """Quetzal's own aggregate leaves base_*_mean blank for sweep-* rows --
    its dump script records the frozen prior's stats once per reward under
    `results/dumps/_base/<reward>/seed{0,42}/dump_summary.json` (`base` key)
    rather than duplicating them onto every config row, unlike MolGPT's/
    G2PT's final_dump which writes base+guided into the same summary.json.
    Read that instead of leaving these bars empty."""
    import glob
    import json
    out = {"parse_rate": [], "uniqueness": []}
    for f in glob.glob(rel("results", "dumps", "_base", "*", "seed*", "dump_summary.json")):
        d = json.load(open(f)).get("base", {})
        for metric in out:
            if metric in d:
                out[metric].append(d[metric])
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}


def model_stats(model):
    rows = load_master(model)
    if model == "Quetzal":
        rows = [r for r in rows if quetzal_ok(r)]
    out = {}
    for source in ("base", "guided"):
        for metric in ("parse_rate", "uniqueness"):
            key = f"{source}_{metric}_mean"
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]
            out[(source, metric)] = (float(np.mean(vals)) if vals else float("nan"), len(vals))
    if model == "Quetzal" and np.isnan(out[("base", "parse_rate")][0]):
        qb = quetzal_base_stats()
        out[("base", "parse_rate")] = (qb["parse_rate"], -1)
        out[("base", "uniqueness")] = (qb["uniqueness"], -1)
    return out


def main():
    for model in ("Quetzal", "MolGPT", "G2PT"):
        require(MASTER_TABLES[model], f"produce {model}'s master_table.csv first")

    models = ["Quetzal", "MolGPT", "G2PT"]
    stats = {m: model_stats(m) for m in models}
    for m in models:
        print(f"[{m}]", {f"{s}_{k}": round(v[0], 4) for (s, k), v in stats[m].items()})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric in zip(axes, ("parse_rate", "uniqueness")):
        x = np.arange(len(models))
        width = 0.35
        base_vals = [stats[m][("base", metric)][0] for m in models]
        guided_vals = [stats[m][("guided", metric)][0] for m in models]
        ax.bar(x - width / 2, base_vals, width, label="base (frozen prior)",
               color=[MODEL_COLOURS[m] for m in models], alpha=0.45)
        ax.bar(x + width / 2, guided_vals, width, label="guided",
               color=[MODEL_COLOURS[m] for m in models], alpha=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.replace("_", " "))

    handles = [plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.45, label="base (frozen prior)"),
               plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.95, label="guided")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False)
    # No suptitle: the caption carries the description.
    fig.tight_layout()
    savefig(fig, "cmp04_validity.png")


if __name__ == "__main__":
    main()
