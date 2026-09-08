#!/usr/bin/env python3
"""Emit tab_capacity.tex: the capacity ladder on Fexofenadine.

Reads the oracle-budget harvest, which evaluates each variant under a fixed oracle-call budget
rather than the fixed post-training sample budget used everywhere else in the dissertation. The
two are not comparable, so every number in this table, including its reference, comes from the
same harvest.

Variants are ordered by trainable-parameter count, from a projection-only update through
low-rank adapters to a full-weight update.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HARVEST = os.path.join(HERE, "..", "oracle_gfn_mols", "_results",
                       "hard_fexofenadine_budget10000.json")
OUT = os.path.join(HERE, "..", "..", "papers", "current", "src", "tab_capacity.tex")

ROWS = [
    ("rtb-proj-fexo-b10-s0",        r"projection only"),
    ("rtb-proj-lora4-fexo-b10-s0",  r"projection $+$ LoRA, rank 4"),
    ("rtb-proj-lora16-fexo-b10-s0", r"projection $+$ LoRA, rank 16"),
    ("rtb-proj-lora64-fexo-b10-s0", r"projection $+$ LoRA, rank 64"),
    ("rtb-atom-fexo-b10-s0",        r"atom stack"),
    ("rtb-atom-lora4-fexo-b10-s0",  r"atom stack $+$ LoRA, rank 4"),
    ("rtb-atom-lora16-fexo-b10-s0", r"atom stack $+$ LoRA, rank 16"),
    ("rtb-atom-lora64-fexo-b10-s0", r"atom stack $+$ LoRA, rank 64"),
    ("rtb-full-fexo-b10-s0",        r"\textbf{all weights}"),
]

CAPTION = (
    r"The capacity ladder on Fexofenadine, under RTB at $\beta=10$. Every variant updates the "
    r"frozen model's own weights rather than training a separate guide, ordered here by how many "
    r"of them it may change. Scores come from the oracle-budget harvest, which is a different "
    r"evaluation protocol from the fixed post-training sample used in \Cref{tab:diss-landscape}, "
    r"so this table is internally comparable but should not be read against that one. The "
    r"GEOM-Drugs reference is the same corpus best-of-10{,}000 measured inside this harvest. "
    r"Bold marks the full-weight update, which is the variant the capacity argument turns on."
)


def main():
    d = json.load(open(HARVEST))
    ref = d["_reference"]
    lines = []
    for key, label in ROWS:
        b = d[key]["budgeted"]
        lines.append(r"%s & %.4f & %.4f & %.4f \\" % (label, b["top1"], b["top10"], b["top100"]))
    tex = (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\caption[The capacity ladder on Fexofenadine]{%s}\n\\label{tab:capacity}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Trainable parameters & top-1 & top-10 & top-100 \\\\\n\\midrule\n"
        "%s\n\\addlinespace\n"
        "GEOM-Drugs, best of 10{,}000 & %.4f & %.4f & %.4f \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        % (CAPTION, "\n".join(lines), ref["top1"], ref["top10"], ref["top100"])
    )
    open(OUT, "w").write(tex)
    print("wrote", OUT)
    print(tex)


if __name__ == "__main__":
    main()
