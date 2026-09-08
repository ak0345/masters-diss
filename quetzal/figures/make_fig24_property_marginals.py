#!/usr/bin/env python3
"""
make_fig24_property_marginals.py -- Figure 24: what guiding does to molecular properties.

A UMAP shows that guided and unguided output overlap. It cannot say which properties moved.
This does, on the same samples, in units the reader can interpret.

Each bar is the guided-minus-prior shift in a physicochemical property, divided by the prior's
own standard deviation for that property. A bar of 1.0 means the guided mean moved by one
prior standard deviation. One panel per frozen architecture.

INPUTS
  results/ablations/chemistry/summary.json   (analyse_chemistry.py)

USAGE
  python figures/make_fig24_property_marginals.py --out out/fig24.pdf
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

LEGS = [("quetzal", "Quetzal"), ("molgpt", "MolGPT"), ("g2pt", "G2PT")]
REWARDS = [("osim", "Osimertinib"), ("peri", "Perindopril"),
           ("fexo", "Fexofenadine"), ("nitrogen", "Nitrogen (dense)")]
RCOL = {"osim": "#4C72B0", "peri": "#DD8452", "fexo": "#55A868", "nitrogen": "#C44E52"}
PROPS = ["logP", "TPSA", "MW", "HBA", "HBD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=fs.rel("results", "ablations", "chemistry", "summary.json"))
    ap.add_argument("--out", default="out/fig24.pdf")
    args = ap.parse_args()
    fs.use_paper_style()
    d = json.load(open(fs.need(args.summary)))

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.2), sharey=True)
    width = 0.2
    for ax, (leg, legname) in zip(axes, LEGS):
        for j, (rk, rname) in enumerate(REWARDS):
            e = d.get(leg, {}).get(rk)
            if not e:
                continue
            shifts = []
            for p in PROPS:
                gm, _ = e["properties"][p]["guided"]
                pm, ps = e["properties"][p]["prior"]
                shifts.append((gm - pm) / ps if ps > 0 else 0.0)
            xs = np.arange(len(PROPS)) + (j - 1.5) * width
            ax.bar(xs, shifts, width, color=RCOL[rk], alpha=0.9,
                   label=rname if leg == "quetzal" else None, zorder=2)
        ax.axhline(0, color="k", lw=0.9, zorder=1)
        ax.set_xticks(range(len(PROPS)))
        ax.set_xticklabels(PROPS, fontsize=8)
        ax.set_title(legname)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("shift in prior std devs")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)
    for leg, _ in LEGS:
        for rk, _ in REWARDS:
            e = d.get(leg, {}).get(rk)
            if not e: continue
            worst = max(PROPS, key=lambda p: abs(
                (e["properties"][p]["guided"][0] - e["properties"][p]["prior"][0])
                / max(e["properties"][p]["prior"][1], 1e-9)))
            gm, _ = e["properties"][worst]["guided"]; pm, ps = e["properties"][worst]["prior"]
            print("[fig24] %-8s %-9s largest shift: %-5s %+.2f sd" % (leg, rk, worst, (gm-pm)/ps))


if __name__ == "__main__":
    main()
