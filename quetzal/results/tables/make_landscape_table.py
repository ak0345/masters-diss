#!/usr/bin/env python3
"""Emit tab_diss_landscape.tex from the aggregated master table.

Pooling rule, applied identically to every cell:
    family == "sweep", beta in {1, 10}, both replay settings, both training
    seeds {0, 42}, grouped by (reward, guide, objective).

The beta=100 stress runs are deliberately excluded here: they exist only for
the hidden guide, so including them would pool a different set of runs into
each row and make the columns non-comparable.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "..", "dumps", "_aggregate", "master_table.csv")
OUT = os.path.join(HERE, "..", "..", "papers", "current", "src", "tab_diss_landscape.tex")

REWARDS = [("osim", "Osimertinib MPO"), ("peri", "Perindopril MPO"),
           ("fexo", "Fexofenadine MPO"), ("nitrogen", "Nitrogen fraction (control)")]
GUIDES = [("base", r"residual (\texttt{LogitGuide})"),
          ("tempgain", r"temp-gain (\texttt{TempGainGuide})"),
          ("hidden", r"hidden (\texttt{HiddenGuide})")]
OBJECTIVES = ["DB", "RTB"]

CAPTION = (
    r"Terminal top-10 score on the guide sweep. Every cell pools $\beta \in \{1,10\}$, "
    r"both replay settings and both training seeds $\{0,42\}$, giving $n=8$ runs per cell. "
    r"The $\beta=100$ stress runs are excluded here because they exist only for the hidden "
    r"guide, and pooling them would put a different set of runs behind each row. The three "
    r"guide architectures are listed in the order they were built (\Cref{sec:guides}); the "
    r"temperature-gain guide was not run on the nitrogen control. Bold marks the nitrogen control's hidden/DB result, the one cell "
    r"in the sweep that is qualitatively different from the frozen prior."
)


def main():
    df = pd.read_csv(MASTER)
    df = df[(df["family"] == "sweep") & (df["beta"].isin([1, 10]))]
    prior = pd.read_csv(MASTER)
    prior = prior[prior["guide"] == "base_prior"].set_index("reward")

    rows = []
    for key, label in REWARDS:
        rows.append(r"\multicolumn{4}{l}{\emph{%s}} \\" % label)
        p = prior.loc[key, "guided_reward_top10_mean"]
        rows.append(r"\quad frozen prior & -- & %.4f & --- \\" % p)
        for gkey, glabel in GUIDES:
            for obj in OBJECTIVES:
                d = df[(df["reward"] == key) & (df["guide"] == gkey)
                       & (df["objective"] == obj.lower())]["guided_reward_top10_mean"]
                if len(d) == 0:
                    rows.append(r"\quad %s & %s & \multicolumn{2}{l}{not run} \\"
                                % (glabel, obj))
                else:
                    mean, sd = d.mean(), d.std()
                    # Bold the one cell that is qualitatively different from the rest of the
                    # sweep: the nitrogen control's hidden/DB result, which the text singles
                    # out. Everything else sits within a few hundredths of the prior.
                    fmt = (r"\quad %s & %s & \textbf{%.4f} & %.4f \\"
                           if (key == "nitrogen" and gkey == "hidden" and obj == "DB")
                           else r"\quad %s & %s & %.4f & %.4f \\")
                    rows.append(fmt % (glabel, obj, mean, sd))
        rows.append(r"\addlinespace")

    body = "\n".join(rows)
    tex = (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\caption[Terminal top-10 on the guide sweep]{%s}\n\\label{tab:diss-landscape}\n"
        "\\begin{tabular}{llrr}\n\\toprule\n"
        "Guide & Objective & top-10 mean & top-10 std \\\\\n\\midrule\n"
        "%s\n\\bottomrule\n\\end{tabular}\n\\end{table}\n" % (CAPTION, body)
    )
    with open(OUT, "w") as fh:
        fh.write(tex)
    print("wrote", OUT)
    print(tex)


if __name__ == "__main__":
    main()
