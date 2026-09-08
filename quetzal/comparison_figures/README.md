# comparison_figures/

Figures and tables that put all three architectures on shared axes. Everything here reads the
per-leg aggregated output, so run the Quetzal, MolGPT and G2PT pipelines first.

Rendered output is not included. Each script takes `--out`.

| script | figure |
|---|---|
| `make_cmp01_landscape.py` | guided top-10 against each model's own corpus ceiling |
| `make_cmp02_positional.py` | flip rate by sequence position, all three legs |
| `make_cmp02b_positional_atomidx.py` | the same, indexed by atom decision rather than token |
| `make_cmp03_delivered_flip.py` | delivered fraction against sample flip rate |
| `make_cmp04_validity.py` | validity and uniqueness, base against guided |
| `make_cmp05_chemspace.py` | per-architecture chemical space on shared axes |
| `make_cmp06_nn_similarity.py` | nearest-neighbour Tanimoto against the unguided null |
| `make_cmp07_bestofn_baseline.py` | guided score against an unguided harvest curve |
| `make_cmp08_fudge_vs_dbrtb.py` | the discriminator's diagnostics against DB and RTB |
| `make_cmp09_fudge_landscape.py` | terminal score, DB and RTB against the discriminator |
| `make_cmp10_chemspace_full.py` | the full joint embedding, every leg and reward |

`compute_geom_baseline.py` builds the shared GEOM-Drugs best-of-10,000 reference that the
corpus-ceiling analysis compares every architecture against. It is computed once from a random
sample of the shared corpus and reused for all three legs, which is what makes the ceiling
comparable across them. `make_comparison_table.py` emits the cross-model summary table.
`figstyle_compare.py` holds the shared palette and per-leg colours.

The two small data files kept here, `geom_baseline_best_of_10k.json` and
`comparison_table.csv`, are the computed reference and summary. They are inputs to the figure
scripts rather than results in their own right, so they are retained to make the figures
runnable without re-running the whole pipeline.
