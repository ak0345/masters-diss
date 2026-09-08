# results/

Generated output lives here and is **not** included in this codebase: no dumps, no diagnostic
reports, no aggregated tables. The pipeline in `scripts/` recreates all of it.

What is kept are the scripts that turn that output into the numbers and tables the manuscript
reports. They resolve their inputs relative to their own location, so they must stay where they
are:

| script | reads | writes |
|---|---|---|
| `tables/make_landscape_table.py` | `results/dumps/_aggregate/master_table.csv` | `tab_diss_landscape.tex` |
| `tables/make_delivery_table.py` | `results/flips-guide/flip_report_*.json` | `tab_diss_delivery.tex` |
| `ablations/sign-test/sign_test.py` | `results/dumps/_aggregate/master_table.csv` | `summary.json` beside itself |
| `ablations/rtb-validity/extract_validity.py` | local wandb run directories | CSV summaries beside itself |
| `ablations/chemistry/analyse_chemistry.py` | guided and prior sample dumps, all three legs | `summary.json` beside itself |
| `ablations/chemistry/separability_per_reward.py` | Quetzal guided dumps | `separability.json` beside itself |

## The chemistry analysis, and the reference it uses

`analyse_chemistry.py` computes five statistics per architecture and reward, each against an
explicit null: the identical-molecule fraction under paired sampling, Bemis-Murcko scaffold
overlap, nearest-neighbour Tanimoto similarity with bootstrap intervals and a permutation test,
and physicochemical property marginals.

Its central design decision is the reference set:

    reference   prior, seed 42
    null        prior seed 0   vs reference
    guided      guided seed 0  vs reference

A guided dump and the prior dump sharing its seed come from one random stream, so many of their
molecules are identical, and an identical molecule scores a maximum Tanimoto of 1.0 against a
reference containing it. Comparing guided against its seed-matched prior while comparing the
null against an independent one is not a like-for-like test, and it produces a large spurious
effect. Both comparisons here use the same independent reference.

The coupling is not discarded: it is reported separately as the identical-molecule fraction,
which measures directly how often the guide changes nothing at all.

`separability_per_reward.py` repeats the pooled k-NN separability test one-versus-rest, since a
pooled result can hide one separable family among several inseparable ones.

The two table scripts write into the manuscript source tree, at
`papers/current/src/`, which is not part of this codebase. Either point `OUT` somewhere else or
place this tree beside the manuscript repository.
