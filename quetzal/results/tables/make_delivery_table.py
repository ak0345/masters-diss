#!/usr/bin/env python3
"""Emit tab_diss_delivery.tex from the raw flip-diagnostic reports.

Pooling rule: one report per swept configuration, beta in {1, 10}, temperature
1.0 block, smoke-test runs excluded. Runs are grouped by the guide tag in their
own run name, NOT by the `guide_type` field: the diagnostic harness instantiates
the residual guide through the TempGainGuide class (the two coincide at T = 1),
so `guide_type` alone would merge two separately trained populations.
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "flips-guide", "flip_report_*.json")
OUT = os.path.join(HERE, "..", "..", "papers", "current", "src", "tab_diss_delivery.tex")

GUIDES = [("base", r"residual (\texttt{LogitGuide})"),
          ("tempgain", r"temp-gain (\texttt{TempGainGuide})"),
          ("hidden", r"hidden (\texttt{HiddenGuide})")]

SHORT_CAPTION = "Delivery and flip diagnostics"
CAPTION = (
    r"Delivery and flip diagnostics by guide architecture, on the main "
    r"$\beta \in \{1,10\}$ sweep, pooled over both training objectives, both replay "
    r"settings, both training seeds $\{0,42\}$ and all four rewards (three for the "
    r"temperature-gain guide, which was not run on the nitrogen control). Figures in "
    r"parentheses are the standard deviation across the pooled runs, not a standard error. "
    r"Here $n$ counts the individual seed-run diagnostic reports averaged over, each itself "
    r"already an average over every evaluated state in that run. High-gap and low-gap flip "
    r"rates split decisions at a frozen-prior top-1 logit margin of 8, which separates "
    r"71.9\,\% of decisions (high-margin) from the rest."
)


def main():
    recs = {tag: [] for tag, _ in GUIDES}
    for path in sorted(glob.glob(REPORTS)):
        doc = json.load(open(path))
        label = str(doc.get("label") or doc.get("ckpt", ""))
        if "smoke" in label:
            continue
        m = re.search(r"sweep-[a-z0-9]+-([a-z]+)-", label)
        b = re.search(r"-b(\d+)-", label)
        if not m or not b or int(b.group(1)) not in (1, 10):
            continue
        blk = doc.get("flip_temp1.0")
        if blk and m.group(1) in recs:
            recs[m.group(1)].append(blk)

    rows = []
    for tag, label in GUIDES:
        R = recs[tag]
        col = lambda k: np.array([r[k] for r in R], dtype=float)
        rows.append(
            r"%s & %d & %.3f & %.3f (%.3f) & %.3f (%.3f) & %.3f (%.3f) & %.4f & %.3f \\"
            % (label, len(R), col("delivered_frac").mean(),
               100 * col("mean_total_variation").mean(),
               100 * col("mean_total_variation").std(),
               100 * col("argmax_flip_rate").mean(), 100 * col("argmax_flip_rate").std(),
               100 * col("sample_flip_rate").mean(), 100 * col("sample_flip_rate").std(),
               100 * col("flip_rate_high_gap").mean(),
               100 * col("flip_rate_low_gap").mean()))

    tex = (
        # Short title first: the list of tables shows it instead of the whole caption.
        "\\begin{table}[t]\n\\centering\\small\n\\caption[%s]{%s}\n"
        "\\label{tab:diss-delivery}\n\\resizebox{\\linewidth}{!}{\n"
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        "Guide & $n$ runs & Delivered & Mean TV \\%% & Argmax flip \\%% & Sample flip \\%% "
        "& High-gap flip \\%% & Low-gap flip \\%% \\\\\n\\midrule\n%s\n"
        "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
        % (SHORT_CAPTION, CAPTION, "\n".join(rows))
    )
    with open(OUT, "w") as fh:
        fh.write(tex)
    print("wrote", OUT)
    print(tex)


if __name__ == "__main__":
    main()
