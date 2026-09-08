#!/usr/bin/env python3
"""
make_fig20_composition.py -- Figure 20: the composition track.

Left: terminal top-10 score of the composed sampler under each mixing operator
(linear, product, harmonic), against the frozen prior and the best single guide
on the same objective. Points are individual composed runs; the bar is the
per-operator mean.

Right: the per-run composed-minus-prior top-10 difference, by operator. The
spread crosses zero for every operator on Osimertinib and sits slightly above it
on Perindopril.

INPUTS
  results/dumps/_aggregate/master_table.csv    (family "compose" and "sweep")

USAGE
  python figures/make_fig20_composition.py --out out/fig20.pdf
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs

OPERATORS = ["linear", "product", "harmonic"]
REWARDS = [("osimertinib", "osim", "Osimertinib MPO"),
           ("perindopril", "peri", "Perindopril MPO")]
OP_COLOUR = {"linear": "#4C72B0", "product": "#DD8452", "harmonic": "#55A868"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=fs.rel("results", "dumps", "_aggregate",
                                               "master_table.csv"))
    ap.add_argument("--out", default="out/fig20.pdf")
    args = ap.parse_args()

    fs.use_paper_style()
    df = pd.read_csv(fs.need(args.master))

    comp = df[df["family"] == "compose"]
    sweep = df[(df["family"] == "sweep") & (df["beta"].isin([1, 10]))]
    prior = df[df["guide"] == "base_prior"].set_index("reward")

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3))

    # ---- panels 1-2: absolute top-10 per reward, zoomed to the data
    for ax, (long_name, short_name, title) in zip(axes[:2], REWARDS):
        c = comp[comp["reward"] == long_name]
        s_ = sweep[sweep["reward"] == short_name]
        p = prior.loc[short_name, "guided_reward_top10_mean"]
        best = s_.groupby(["guide", "objective"])["guided_reward_top10_mean"].mean().max()
        vals = []
        for i, op in enumerate(OPERATORS):
            d = c[c["operator"] == op]["guided_reward_top10_mean"].values
            vals.append(d)
            ax.scatter(np.full(len(d), i), d, s=22, color=OP_COLOUR[op],
                       alpha=0.85, zorder=3)
            ax.hlines(d.mean(), i - 0.3, i + 0.3, color="k", lw=1.6, zorder=4)
        ax.axhline(p, color=fs.REF_COLOUR, lw=1.4, zorder=2, label="frozen prior")
        ax.axhline(best, color="k", lw=1.2, ls="--", zorder=2,
                   label="best single guide")
        lo = min(float(np.min(v)) for v in vals + [np.array([p, best])])
        hi = max(float(np.max(v)) for v in vals + [np.array([p, best])])
        pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(len(OPERATORS)))
        ax.set_xticklabels(OPERATORS, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("terminal top-10 score")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    # ---- panel 3: composed-minus-prior difference, both rewards
    ax = axes[2]
    xs, labels = [], []
    pos = 0
    for long_name, short_name, title in REWARDS:
        c = comp[comp["reward"] == long_name]
        for op in OPERATORS:
            d = c[c["operator"] == op]["top10_delta_mean"].values
            ax.scatter(np.full(len(d), pos), d, s=18, color=OP_COLOUR[op],
                       alpha=0.85, zorder=3)
            ax.hlines(d.mean(), pos - 0.3, pos + 0.3, color="k", lw=1.5, zorder=4)
            xs.append(pos); labels.append(op)
            pos += 1
        pos += 1
    ax.axhline(0.0, color="k", lw=0.9, alpha=0.6, zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("composed $-$ prior top-10")
    ax.set_title("Per-run difference")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)
    for long_name, _, _ in REWARDS:
        c = comp[comp["reward"] == long_name]
        print("[fig] %s composed mean %.4f, %d/%d runs above prior"
              % (long_name, c["guided_reward_top10_mean"].mean(),
                 int((c["top10_delta_mean"] > 0).sum()), len(c)))


if __name__ == "__main__":
    main()
