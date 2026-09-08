"""make_cmp03_delivered_flip.py -- three-way comparison: how much of each
guide's designed reach actually registers, by model.

Grouped bars, one group per model, for three flip-diagnostic metrics
(delivered_frac, argmax_flip_rate, sample_flip_rate), pooled (mean) across
every config in that model's own pilot-comparable grid (Quetzal: sweep-*,
guide in {base,hidden}, objective in {db,rtb}, replay=off, beta=10; MolGPT
and G2PT: their whole 48-config grid, which is already exactly this scope).

G2PT's numbers use the `atom_steps` regime (see flip_g2pt.py / decisions.md,
"G2PT leg uses Option A") -- the regime where its guide is actually active
under step-gating, which is the fair comparison point against Quetzal's and
MolGPT's guides (dimension-masked at every position, no step-gating, so
their one number already is this).
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (flip_reports, flip_metric, savefig, require,
                               plt, MODEL_COLOURS, FLIPS_DIRS, NAME_RE)

METRICS = ["delivered_frac", "argmax_flip_rate", "sample_flip_rate"]
METRIC_TITLE = {"delivered_frac": "delivered fraction",
                 "argmax_flip_rate": "argmax flip rate",
                 "sample_flip_rate": "sample flip rate"}


def quetzal_ok(label):
    """sweep-* only (excludes compose-* composition-ablation runs, no
    MolGPT/G2PT equivalent), guide in {base,hidden} (excludes tempgain),
    replay off, beta 10 -- the exact grid the pilots share with Quetzal."""
    if not label.startswith("sweep-"):
        return False
    m = NAME_RE.match(label)
    return bool(m) and m.group("guide") in ("base", "hidden") \
        and m.group("replay") == "off" and m.group("beta") == "10"


def model_means(model):
    reports = flip_reports(model)
    if model == "Quetzal":
        reports = {k: v for k, v in reports.items() if quetzal_ok(k)}
    out = {}
    for metric in METRICS:
        vals = [flip_metric(r, model, metric) for r in reports.values()]
        vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
        out[metric] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return out


def main():
    for model in ("Quetzal", "MolGPT", "G2PT"):
        require(FLIPS_DIRS[model], f"produce {model}'s flip_report_*.json first")

    models = ["Quetzal", "MolGPT", "G2PT"]
    stats = {m: model_means(m) for m in models}
    for m in models:
        print(f"[{m}]", {k: round(v[0], 4) for k, v in stats[m].items()},
              "n=", {k: v[2] for k, v in stats[m].items()})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    x = np.arange(len(METRICS))
    for i, model in enumerate(models):
        means = [stats[model][m][0] for m in METRICS]
        stds = [stats[model][m][1] for m in METRICS]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
               color=MODEL_COLOURS[model], label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_TITLE[m] for m in METRICS])
    ax.set_ylabel("mean across all configs (± std)")
    ax.set_title("Delivery against flip rate")
    ax.legend(frameon=False)
    savefig(fig, "cmp03_delivered_flip.png")


if __name__ == "__main__":
    main()
