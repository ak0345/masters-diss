#!/usr/bin/env python3
"""
make_fig25_peri_weights.py -- Figure 25: the Perindopril composition-weight sweep.

Perindopril's objective has two live components, so the weight simplex is a line and
eleven points of omega_0, with omega_1 = 1 - omega_0, cover it completely at each of the
three mixing operators.

  A  the two component sub-scores against each other, one point per (omega, operator).
     A trade-off would show as a downward-sloping frontier.
  B  terminal top-10 against omega, with the frozen prior's own top-10 range as a band.
  C  nearest-neighbour Tanimoto to an INDEPENDENT prior draw against omega, with the
     matched unguided-prior control band. This is the reachable-space measure in chemistry:
     a point leaving the band would be a sample sitting somewhere the prior does not go.

  D  hypervolume of the attained (c0, c1) front against omega. The same question in
     objective space. Recomputed over the two real objectives with one reference point for
     every point in the sweep, because the dumps' own hypervolume field carries a redundant
     third axis and refits its score floors per invocation.

INPUTS
  results/ablations/peri-weights/summary.json   (scripts/analyse_peri_weights.py)

USAGE
  python figures/make_fig25_peri_weights.py --out out/fig25_peri_weights.pdf
"""
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

OPERATORS = [("linear", "linear", "o"), ("product", "product", "s"),
             ("harmonic", "harmonic", "^")]
COL = {"linear": "#4C72B0", "product": "#DD8452", "harmonic": "#55A868"}


def by_op(points, op):
    return sorted([p for p in points if p["operator"] == op], key=lambda p: p["w0"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary",
                    default=fs.rel("results", "ablations", "peri-weights", "summary.json"))
    ap.add_argument("--out", default="out/fig25_peri_weights.pdf")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    fs.use_paper_style()

    d = json.load(open(fs.need(args.summary)))
    pts = d["points"]
    ctrl = d["prior_chunk_control"]

    # 2x2 rather than a single row of 4: at \linewidth in the manuscript a
    # single row this wide shrank axis and tick labels below legibility.
    fig, axes2d = plt.subplots(2, 2, figsize=(8.8, 7.2))
    axes = [axes2d[0, 0], axes2d[0, 1], axes2d[1, 0], axes2d[1, 1]]

    # ---- A: the two sub-scores against each other -------------------------
    ax = axes[0]
    sc = None
    for op, label, marker in OPERATORS:
        ps = by_op(pts, op)
        sc = ax.scatter([p["c0"] for p in ps], [p["c1"] for p in ps],
                        c=[p["w0"] for p in ps], cmap="viridis", vmin=0.0, vmax=1.0,
                        marker=marker, s=34, edgecolors="k", linewidths=0.4, zorder=3)
    ax.scatter([pts[0]["c0_base"]], [pts[0]["c1_base"]], marker="*", s=150,
               color="#C44E52", edgecolors="k", linewidths=0.5, zorder=4)
    # Shape carries the operator, colour carries omega, so the legend must not take its
    # colours from the colourmap.
    handles = [plt.Line2D([0], [0], ls="", marker=m, mfc="0.75", mec="k", mew=0.4,
                          ms=5, label=lab) for _, lab, m in OPERATORS]
    handles.append(plt.Line2D([0], [0], ls="", marker="*", mfc="#C44E52", mec="k",
                              mew=0.5, ms=10, label="frozen prior"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"$\omega_0$", fontsize=10.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel(r"component 0 sub-score")
    ax.set_ylabel(r"component 1 sub-score")
    ax.set_title("A. Sub-scores", fontsize=10.5)

    # ---- B: top-10 against omega, with the prior band ---------------------
    ax = axes[1]
    lo, hi = d["top10_base_range"]
    ax.axhspan(lo, hi, color=fs.BACKDROP_COLOURS["prior"], alpha=0.20, zorder=1)
    ax.text(0.02, hi, " frozen prior", va="bottom", ha="left", fontsize=8,
            color="0.3", transform=ax.get_yaxis_transform(which="grid"))
    for op, label, marker in OPERATORS:
        ps = by_op(pts, op)
        ax.plot([p["w0"] for p in ps], [p["top10"] for p in ps],
                marker=marker, ms=4, color=COL[op], label=label, zorder=3)
    ax.set_xlabel(r"$\omega_0$   ($\omega_1 = 1-\omega_0$)")
    ax.set_ylabel("top-10 log reward")
    ax.set_title("B. Score", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8)

    # ---- C: reachable space, against the matched control ------------------
    ax = axes[2]
    ax.axhspan(ctrl["nn_range"][0], ctrl["nn_range"][1],
               color=fs.BACKDROP_COLOURS["prior"], alpha=0.20, zorder=1)
    ax.axhline(ctrl["nn_to_independent_prior_mean"], color="0.35", lw=1.0, ls="--", zorder=2)
    ax.text(0.02, ctrl["nn_range"][1], " unguided prior", va="bottom", ha="left",
            fontsize=8, color="0.3", transform=ax.get_yaxis_transform(which="grid"))
    for op, label, marker in OPERATORS:
        ps = by_op(pts, op)
        ax.plot([p["w0"] for p in ps], [p["nn_to_independent_prior"] for p in ps],
                marker=marker, ms=4, color=COL[op], label=label, zorder=3)
    ax.set_xlabel(r"$\omega_0$   ($\omega_1 = 1-\omega_0$)")
    ax.set_ylabel("NN Tanimoto to independent prior")
    ax.set_title("C. Reachable space", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8)

    # ---- D: hypervolume of the attained front ----------------------------
    # No matched-prior band here. The per-point composed-versus-prior comparison was dropped:
    # the 33 configurations collapse to 4 distinct populations, so any test over them counts
    # one outcome many times. The panel shows the flatness across omega, which is the finding.
    ax = axes[3]
    for op, label, marker in OPERATORS:
        ps = by_op(pts, op)
        ax.plot([p["w0"] for p in ps], [p["hv_composed"] for p in ps],
                marker=marker, ms=4, color=COL[op], label=label, zorder=3)
    ax.set_xlabel(r"$\omega_0$   ($\omega_1 = 1-\omega_0$)")
    ax.set_ylabel("hypervolume of the (c0, c1) front")
    ax.set_title("D. Objective space", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fs.save(fig, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
