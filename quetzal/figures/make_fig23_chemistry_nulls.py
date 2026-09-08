#!/usr/bin/env python3
"""
make_fig23_chemistry_nulls.py -- Figure 23: chemistry-space statistics against their nulls.

Three statistics that a UMAP projection cannot supply, each with the null it has to be read
against.

  A  mean maximum Tanimoto similarity to an independent draw of the frozen prior, guided
     against the prior-vs-prior null, with bootstrap 95% intervals.
  B  Bemis-Murcko scaffold overlap (Jaccard) against the same null.
  C  the fraction of molecules the guide leaves completely unchanged under paired sampling,
     which is only defined where guided and prior dumps share a sampling seed.

INPUTS
  results/ablations/chemistry/summary.json   (analyse_chemistry.py)

USAGE
  python figures/make_fig23_chemistry_nulls.py --out out/fig23.pdf
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

LEGS = [("quetzal", "Quetzal"), ("molgpt", "MolGPT"), ("g2pt", "G2PT")]
REWARDS = [("osim", "Osim"), ("peri", "Peri"), ("fexo", "Fexo"), ("nitrogen", "Nitrogen")]
COL = {"quetzal": "#4C72B0", "molgpt": "#DD8452", "g2pt": "#55A868"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=fs.rel("results", "ablations", "chemistry", "summary.json"))
    ap.add_argument("--out", default="out/fig23.pdf")
    args = ap.parse_args()
    fs.use_paper_style()
    d = json.load(open(fs.need(args.summary)))

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4))

    # ---- A: NN similarity with bootstrap CIs
    ax = axes[0]; pos = 0; ticks = []; labels = []
    for leg, legname in LEGS:
        for rk, rname in REWARDS:
            e = d.get(leg, {}).get(rk)
            if not e: continue
            g, lo, hi = e["nn"]["guided_mean"], *e["nn"]["guided_ci"]
            n = e["nn"]["null_mean"]
            ax.errorbar(pos, g, yerr=[[g - lo], [hi - g]], fmt="o", ms=4,
                        color=COL[leg], capsize=2, zorder=3)
            ax.hlines(n, pos - 0.34, pos + 0.34, color="k", lw=1.2, zorder=4)
            if e["nn"]["p_perm"] < 0.05:
                ax.text(pos, hi + 0.0015, "*", ha="center", fontsize=9)
            ticks.append(pos); labels.append(rname); pos += 1
        pos += 0.8
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("mean max Tanimoto to prior")
    ax.set_title("A. Similarity to the prior")

    # ---- B: scaffold Jaccard against null
    ax = axes[1]; pos = 0; ticks = []; labels = []
    for leg, legname in LEGS:
        for rk, rname in REWARDS:
            e = d.get(leg, {}).get(rk)
            if not e: continue
            ax.scatter(pos, e["scaffold"]["guided"]["jaccard"], s=26, color=COL[leg], zorder=3)
            ax.hlines(e["scaffold"]["null"]["jaccard"], pos - 0.34, pos + 0.34,
                      color="k", lw=1.2, zorder=4)
            ticks.append(pos); labels.append(rname); pos += 1
        pos += 0.8
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("scaffold Jaccard vs prior")
    ax.set_title("B. Scaffold overlap")

    # ---- C: identical-molecule fraction, paired sampling (Quetzal only)
    ax = axes[2]
    ks = [(rk, rname) for rk, rname in REWARDS if rk in d.get("quetzal", {})]
    vals = [100 * d["quetzal"][rk]["identical_frac"] for rk, _ in ks]
    ax.bar(range(len(ks)), vals, 0.6, color=COL["quetzal"], alpha=0.85, zorder=2)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([n for _, n in ks], rotation=45,
                                                      ha="right", fontsize=7)
    ax.set_ylabel("molecules left unchanged (\\%)" if False else "molecules left unchanged (%)")
    ax.set_title("C. Unchanged under paired sampling")

    for a in axes:
        a.grid(alpha=0.3, axis="y")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COL[k], label=v) for k, v in LEGS]
    handles.append(plt.Line2D([], [], color="k", lw=1.2, label="null (prior vs prior)"))
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="lower left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
