"""make_cmp08_fudge_vs_dbrtb.py -- cross-model comparison of the two guide
training objectives now available for every leg: the original GFlowNet
DB/RTB guides vs. FUDGE (masked binary cross-entropy on a realized-token
label -- see papers/current/decisions.md's "FUDGE baseline" entries). Same
underlying guide architecture in both cases (`LogitGuide`, h -> vocab_size,
zero-init last layer) -- the only thing that differs between the two bars
per model is the training objective, which is exactly the point of this
comparison: does switching objectives change how much of the guide's reach
actually registers?

DB/RTB numbers: mean across each model's own pilot-comparable grid, same
scope and pooling as make_cmp03_delivered_flip.py (reused directly).
FUDGE numbers: mean across the 4 fudge-<reward> flip reports per model
(one training objective, one guide, no beta/replay/seed axis to pool over).
G2PT uses the `atom_steps` regime for both, matching cmp03's own reasoning
(the regime where G2PT's guide is actually active under step-gating).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (rel, flip_reports, t1_block, savefig, require,
                               plt, json, MODEL_COLOURS, FLIPS_DIRS, NAME_RE)
from make_cmp03_delivered_flip import METRICS, METRIC_TITLE, quetzal_ok

FUDGE_FLIPS_DIRS = {
    "Quetzal": rel("results", "fudge_flips"),
    "MolGPT": rel("molgpt", "results", "molgpt", "fudge_flips"),
    "G2PT": rel("g2pt", "results", "g2pt", "fudge_flips"),
}


def dbrtb_means(model):
    reports = flip_reports(model)
    if model == "Quetzal":
        reports = {k: v for k, v in reports.items() if quetzal_ok(k)}
    out = {}
    for metric in METRICS:
        vals = [t1_block(model, r).get(metric) for r in reports.values()]
        vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
        out[metric] = (float(np.mean(vals)) if vals else np.nan, len(vals))
    return out


def fudge_means(model):
    reports = [json.load(open(f)) for f in
               sorted(glob.glob(f"{FUDGE_FLIPS_DIRS[model]}/flip_report_fudge-*.json"))]
    out = {}
    for metric in METRICS:
        vals = [t1_block(model, r).get(metric) for r in reports]
        vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
        out[metric] = (float(np.mean(vals)) if vals else np.nan, len(vals))
    return out


def main():
    for model in ("Quetzal", "MolGPT", "G2PT"):
        require(FLIPS_DIRS[model], f"produce {model}'s DB/RTB flip_report_*.json first")
        require(FUDGE_FLIPS_DIRS[model], f"produce {model}'s FUDGE flip_report_fudge-*.json first")

    models = ["Quetzal", "MolGPT", "G2PT"]
    dbrtb = {m: dbrtb_means(m) for m in models}
    fudge = {m: fudge_means(m) for m in models}
    for m in models:
        print(f"[{m}] DB/RTB", {k: round(v[0], 4) for k, v in dbrtb[m].items()},
              "n=", {k: v[1] for k, v in dbrtb[m].items()})
        print(f"[{m}] FUDGE ", {k: round(v[0], 4) for k, v in fudge[m].items()},
              "n=", {k: v[1] for k, v in fudge[m].items()})

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.5))
    for ax, metric in zip(axes, METRICS):
        x = np.arange(len(models))
        width = 0.35
        db_vals = [dbrtb[m][metric][0] for m in models]
        fu_vals = [fudge[m][metric][0] for m in models]
        ax.bar(x - width / 2, db_vals, width, color=[MODEL_COLOURS[m] for m in models],
               alpha=0.45, label="DB/RTB")
        ax.bar(x + width / 2, fu_vals, width, color=[MODEL_COLOURS[m] for m in models],
               alpha=0.95, label="FUDGE")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(METRIC_TITLE[metric])

    handles = [plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.45, label="DB/RTB"),
               plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.95, label="FUDGE")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False)
    # No suptitle: the caption carries the description.
    fig.tight_layout()
    savefig(fig, "cmp08_fudge_vs_dbrtb.png")


if __name__ == "__main__":
    main()
