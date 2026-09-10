#!/usr/bin/env python3
"""Sign of the guided-vs-prior effect, by reward, at the level of independent design cells.

Supersedes `results/ablations/sign-test/sign_test.py`'s paired sign test, which treated each
of the up to 48 swept configurations (guide architecture x objective x beta x replay x seed)
as an independent trial. Withdrawn as a significance test 2026-09-09
(papers/current/decisions.md): within one (guide, objective) cell, the 8 configurations that
vary only in beta/replay/seed agree in direction almost unanimously, which is not what
independent trials look like. This script reports only the count of cells whose configurations
favour the guide and does not compute a p-value from it.

Each cell is one (guide architecture, objective) pair, pooling its own beta/replay/seed
configurations by majority sign of `guided_reward_top10_mean - prior`, where `prior` is the
single reward-level frozen-prior top-10 Table 3.1 itself reports (the `base_prior` row for
that reward), not each run's own reused base dump. An earlier version of this script compared
against the per-run reused base instead (`top10_delta_mean`, computed in log-reward space
against whichever base-dump instance final_dump.py happened to reuse for that run); that base
dump varies by dump seed and carries real sampling noise of its own -- as much as 0.03 in
log-reward top-10 for a single reward -- so a run could show a negative sign against its own
noisy reused base while its guided score still sat above Table 3.1's single pooled reference.
That produced counts inconsistent with Table 3.1 for two of the four rewards (2026-09-09,
papers/current/decisions.md). Comparing every run in a cell against the one prior value the
table itself reports removes that inconsistency by construction.

Reads the same aggregated master table the original test did; `results/` itself is
write-guarded read-only evidence, so this new script lives under `scripts/` per the
workspace convention and writes its own summary alongside it.

USAGE
  python scripts/sign_test_cells.py
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "..", "results", "dumps", "_aggregate", "master_table.csv")
REWARDS = ["osim", "peri", "fexo", "nitrogen"]


def main():
    df = pd.read_csv(MASTER)
    prior = (df[df["guide"] == "base_prior"]
             .set_index("reward")["guided_reward_top10_mean"].to_dict())
    # Keep only the single-guide sweep. "base_prior" is the unguided reference, not a
    # paired observation; "base" IS a trained guide (the residual/LogitGuide architecture)
    # and belongs in the count. Compositional runs live in family "compose" and are counted
    # separately, since they pool per-component guides rather than one trained guide.
    df = df[df["family"] == "sweep"]
    df = df[df["guide"] != "base_prior"]
    # Restrict to the main sweep, the same pooling rule Table 3.1 uses. The beta=100 runs
    # exist only for the hidden guide, so including them would weight one architecture more
    # heavily than the other two.
    df = df[df["beta"].isin([1, 10])]
    df = df[df["guided_reward_top10_mean"].notna()]

    out = {}
    for reward in REWARDS:
        p = prior.get(reward)
        d = df[df["reward"] == reward]
        cells = []
        for (guide, objective), g in d.groupby(["guide", "objective"]):
            favouring = int((g["guided_reward_top10_mean"] > p).sum())
            cells.append({"guide": guide, "objective": objective,
                          "favouring_guide": favouring, "n": int(len(g))})
        if not cells:
            continue
        cells_favouring = sum(1 for c in cells if c["favouring_guide"] > c["n"] / 2)
        out[reward] = {
            "cells_favouring_guide": cells_favouring,
            "n_cells": len(cells),
            "prior": p,
            "cells": cells,
        }

    out_path = os.path.join(HERE, "sign_test_cells_summary.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
