#!/usr/bin/env python3
"""Paired sign test: does guiding beat the frozen prior across the swept configurations?

Each swept configuration contributes one paired observation, its own guided-minus-prior
top-10 difference (the ``top10_delta_mean`` column of the aggregated master table, which is
already computed against the matching frozen-prior run for the same reward and seed).
The null is that guiding is as likely to hurt as to help, so the count of positive
differences is Binomial(n, 1/2) and the reported p-value is one-sided.

Writes summary.json next to this file.
"""
import json
import os
from math import comb

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "..", "..", "dumps", "_aggregate", "master_table.csv")
REWARDS = ["osim", "peri", "fexo", "nitrogen"]


def one_sided_binomial(k, n):
    """P(X >= k) for X ~ Binomial(n, 1/2)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n


def main():
    df = pd.read_csv(MASTER)
    # Keep only the single-guide sweep. "base_prior" is the unguided reference, not a
    # paired observation; "base" IS a trained guide (the residual/LogitGuide architecture)
    # and belongs in the test. Compositional runs live in family "compose" and are tested
    # separately, since they pool per-component guides rather than one trained guide.
    df = df[df["family"] == "sweep"]
    df = df[df["guide"] != "base_prior"]
    # Restrict to the main sweep, the same pooling rule Table 3.1 uses. The beta=100 runs
    # exist only for the hidden guide, so including them would weight one architecture more
    # heavily than the other two.
    df = df[df["beta"].isin([1, 10])]
    df = df[df["top10_delta_mean"].notna()]

    out = {}
    pooled_k = pooled_n = 0
    for reward in REWARDS:
        d = df[df["reward"] == reward]["top10_delta_mean"]
        k = int((d > 0).sum())
        n = int(len(d))
        if n == 0:
            continue
        out[reward] = {"k": k, "n": n, "p": round(one_sided_binomial(k, n), 4)}
        pooled_k += k
        pooled_n += n

    out["pooled"] = {
        "k": pooled_k,
        "n": pooled_n,
        "p": round(one_sided_binomial(pooled_k, pooled_n), 5),
    }
    # Bonferroni across the four reward families at a 0.05 family-wise rate.
    out["bonferroni_threshold"] = 0.05 / len(REWARDS)

    with open(os.path.join(HERE, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
