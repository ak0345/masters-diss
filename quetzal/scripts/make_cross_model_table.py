#!/usr/bin/env python3
"""make_cross_model_table.py -- emits tab_cross_model.tex from
comparison_figures/comparison_table.csv (itself produced by
comparison_figures/make_comparison_table.py, which already reads G2PT's
flip metrics at the `atom_steps` regime -- the actual analogue of the other
two architectures' single delivered_frac).

Rerun make_comparison_table.py first if the underlying flip reports or
master tables have changed.

USAGE
  python scripts/make_cross_model_table.py \
      --out papers/current/src/tab_cross_model.tex
"""
import argparse
import csv
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARD_ORDER = ["nitrogen", "osim", "peri", "fexo"]
REWARD_LABEL = {"nitrogen": "Nitrogen (control)", "osim": "Osimertinib MPO",
                "peri": "Perindopril MPO", "fexo": "Fexofenadine MPO"}
MODEL_ORDER = ["Quetzal", "MolGPT", "G2PT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        REPO_ROOT, "comparison_figures", "comparison_table.csv"))
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "papers", "current", "src", "tab_cross_model.tex"))
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["model"], r["reward"]): r for r in rows}

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\caption[Cross-architecture summary]{Cross-architecture summary, computed "
                 r"identically across the three architectures. $n$ "
                 r"configs is the number of guide configurations pooled per cell (DB/RTB "
                 r"$\times$ replay on/off for Quetzal's matched subset; DB/RTB only, no "
                 r"replay sweep, for MolGPT and G2PT). Guided top-10 is on the reward's own "
                 r"$[0,1]$ GuacaMol scale. The GEOM best-of-10k column is the same fixed "
                 r"reference for all three architectures, sampled once from the shared "
                 r"training corpus. Delivered and argmax flip read G2PT's "
                 r"\texttt{atom\_steps} regime, the actual analogue of the other two "
                 r"architectures' single delivered fraction.}")
    lines.append(r"\label{tab:cross-model}")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Reward & $n$ configs & Guided parse & Guided top-10 & Delivered & "
                 r"Argmax flip & GEOM best-of-10k \\")
    lines.append(r"\midrule")
    for model in MODEL_ORDER:
        for reward in REWARD_ORDER:
            r = by_key.get((model, reward))
            if not r:
                continue
            lines.append(
                f"{model} & {REWARD_LABEL[reward]} & {r['n_configs']} & "
                f"{float(r['guided_parse_rate']):.3f} & {float(r['guided_top10_score']):.4f} & "
                f"{float(r['delivered_frac']):.3f} & {100*float(r['argmax_flip_rate']):.3f}\\% & "
                f"{float(r['geom_best_of_10k_top10']):.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
