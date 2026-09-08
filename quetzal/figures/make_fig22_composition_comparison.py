#!/usr/bin/env python3
"""
make_fig22_composition_comparison.py -- Figure 22: composition placed alongside
the three single-objective guide architectures.

Figure 20 compares the composed sampler against the frozen prior. This figure
puts it on the same axes as the guides it is meant to improve on, restricted to
the two benchmarks the composition track covers and to beta = 10, replay off,
which is the setting the component guides were trained at.

  A  guided-minus-prior top-10 difference, per run, by design.
  B  FCD between each design's own output and its matched unguided draw. A
     larger value means the design's output distribution sits further from the
     frozen prior's, whether or not that buys any score.

INPUTS
  results/dumps/_aggregate/master_table.csv   (families "sweep" and "compose")

USAGE
  python figures/make_fig22_composition_comparison.py --out out/fig22.pdf
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs

REWARDS = [("osim", "osimertinib", "Osimertinib MPO", "#4C72B0"),
           ("peri", "perindopril", "Perindopril MPO", "#DD8452")]
DESIGNS = [("base", "residual"), ("tempgain", "temp-gain"), ("hidden", "hidden"),
           ("linear", "comp: linear"), ("product", "comp: product"),
           ("harmonic", "comp: harmonic")]


def series(df_sweep, df_comp, short, long_name, key, column):
    """Per-run values for one design on one benchmark."""
    if key in ("base", "tempgain", "hidden"):
        d = df_sweep[(df_sweep["reward"] == short) & (df_sweep["guide"] == key)]
    else:
        d = df_comp[(df_comp["reward"] == long_name) & (df_comp["operator"] == key)]
    return d[column].dropna().values


def panel(ax, df_sweep, df_comp, column, ylabel, title, zero_line):
    width = 0.38
    for j, (short, long_name, label, colour) in enumerate(REWARDS):
        xs = np.arange(len(DESIGNS)) + (j - 0.5) * width
        for x, (key, _) in zip(xs, DESIGNS):
            v = series(df_sweep, df_comp, short, long_name, key, column)
            if len(v) == 0:
                continue
            ax.scatter(np.full(len(v), x), v, s=13, color=colour, alpha=0.7, zorder=3)
            ax.hlines(v.mean(), x - 0.16, x + 0.16, color="k", lw=1.5, zorder=4)
        ax.scatter([], [], s=22, color=colour, label=label)
    if zero_line:
        ax.axhline(0.0, color="k", lw=0.9, alpha=0.6, zorder=1)
    ax.set_xticks(np.arange(len(DESIGNS)))
    ax.set_xticklabels([lab for _, lab in DESIGNS], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=fs.rel("results", "dumps", "_aggregate",
                                               "master_table.csv"))
    ap.add_argument("--out", default="out/fig22.pdf")
    args = ap.parse_args()

    fs.use_paper_style()
    df = pd.read_csv(fs.need(args.master))
    # beta = 10, replay off: the setting the component guides were trained at, so
    # the single-guide rows here are the matched comparison rather than the whole sweep.
    sweep = df[(df["family"] == "sweep") & (df["beta"] == 10)
               & (df["replay"] == "off")]
    comp = df[df["family"] == "compose"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5))
    panel(axes[0], sweep, comp, "top10_delta_mean",
          "guided $-$ prior top-10", "Score gain over the prior", True)
    panel(axes[1], sweep, comp, "fcd_guided_vs_base_mean",
          "FCD to matched unguided draw", "Distance from the prior", False)
    axes[0].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)
    for short, long_name, label, _ in REWARDS:
        for key, name in DESIGNS:
            d = series(sweep, comp, short, long_name, key, "top10_delta_mean")
            f = series(sweep, comp, short, long_name, key, "fcd_guided_vs_base_mean")
            if len(d):
                print("[fig22] %-16s %-14s delta %+.4f (n=%d)  FCD %.3f"
                      % (label, name, d.mean(), len(d),
                         float(np.nanmean(f)) if len(f) else float("nan")))


if __name__ == "__main__":
    main()
