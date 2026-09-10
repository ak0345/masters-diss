"""make_fig01_landscape.py -- terminal log-reward, guided vs base, one point
per configuration, coloured by guide architecture, faceted by reward.

Adapted from the parent repo's make_fig01_landscape.py (same "one point per
config against its own base" idea). Reference lines added 2026-09-05: the
shared GEOM-Drugs best-of-10k score (same filtered corpus both new legs
train on, see ../../shared_data/PROVENANCE.md) and the published GuacaMol
baselines (Brown et al. 2019, osim/peri/fexo only -- no published number
exists for the nitrogen-fraction control). Both are BEST-OF-10 statistics,
log-transformed to sit on this panel's MEAN-log-reward axis -- a mixed axis,
same deliberate choice Quetzal's own fig01 makes ("the axis is mixed, and
deliberately so"), with the two kinds of quantity kept visually distinct
(scatter points vs horizontal reference lines) rather than pretending
they're the same statistic.

Reads results/molgpt/_aggregate/master_table.csv (`08_aggregate.sh`'s output).

The y-axis label carries "MolGPT" as a short prefix. Two copies of this figure (this one and
g2pt/figures/make_fig01_landscape.py's) are stacked in the dissertation's Figure 4.1, and
without an in-image identifier a reader has to hold the caption's "MolGPT above, G2PT below"
in mind while reading the panels themselves. This is the shortest identifier that says which
architecture without adding a suptitle the caption would then duplicate.
"""
import csv
import math
import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_pilot import (MASTER_TABLE, GUIDE_COLOURS, REWARD_TITLE, savefig,
                             require, plt, json, PUBLISHED, GEOM_BASELINE_PATH)

ARCH = "MolGPT"


def main():
    require(MASTER_TABLE, "bash scripts_molgpt/08_aggregate.sh")
    rows = list(csv.DictReader(open(MASTER_TABLE)))
    if not rows:
        raise SystemExit(f"[FATAL] {MASTER_TABLE} has no rows")
    geom_baseline = json.load(open(GEOM_BASELINE_PATH)) if os.path.exists(GEOM_BASELINE_PATH) else {}

    rewards = sorted({r["reward"] for r in rows})
    fig, axes = plt.subplots(1, len(rewards), figsize=(4 * len(rewards), 4), sharey=True)
    if len(rewards) == 1:
        axes = [axes]

    for ax, reward in zip(axes, rewards):
        gb = geom_baseline.get(reward, {}).get("top10")
        if gb:
            ax.axhline(math.log(gb), color="0.4", ls="--", lw=1.1, zorder=1,
                       label="GEOM-Drugs best-of-10k (top-10)")
        for pub_label, val in PUBLISHED.get(reward, {}).items():
            ax.axhline(math.log(val), color="0.6", ls=":", lw=1.0, zorder=1)
        sub = [r for r in rows if r["reward"] == reward]
        for guide in ("base", "hidden"):
            gsub = [r for r in sub if r["guide"] == guide]
            xs = list(range(len(gsub)))
            base_y = [float(r["base_log_reward_mean_mean"]) for r in gsub if r["base_log_reward_mean_mean"]]
            guided_y = [float(r["guided_log_reward_mean_mean"]) for r in gsub if r["guided_log_reward_mean_mean"]]
            labels = [f"{r['objective']}" for r in gsub]
            colour = GUIDE_COLOURS.get(guide, "#333333")
            if guided_y:
                ax.scatter(xs[:len(guided_y)], guided_y, color=colour, marker="o",
                           label=f"{guide} (guided)", zorder=3)
            if base_y:
                ax.scatter(xs[:len(base_y)], base_y, color=colour, marker="x",
                           label=f"{guide} (base)", alpha=0.6, zorder=2)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(REWARD_TITLE.get(reward, reward))
        ax.set_xlabel("objective")
    axes[0].set_ylabel(f"{ARCH}\nterminal log-reward (mean, N=5000)")
    axes[0].legend(fontsize=7, loc="best")
    # No suptitle: the caption carries the description.
    savefig(fig, "fig01_landscape.pdf")


if __name__ == "__main__":
    main()
