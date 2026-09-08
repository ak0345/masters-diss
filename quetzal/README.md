# quetzal/

The main study. A frozen Quetzal generator, a small guide trained on top of it, and the
diagnostics that measure what the guide actually does.

Everything here imports by bare module name from this directory, so run from here or go through
`scripts/common.sh`. See the top-level `README.md` for setup and the full pipeline.

## Subdirectories

| directory | what it holds |
|---|---|
| `scripts/` | the numbered pipeline, stage 1 to stage 9. Start at `scripts/README.md`. |
| `ablations/` | mechanism probes: flip diagnostics, margin ceilings, atoms-vs-coordinates |
| `figures/` | figure scripts for the Quetzal results |
| `comparison_figures/` | cross-architecture figure scripts, all three legs together |
| `results/` | generated output is excluded; the scripts that read it are kept |
| `notebooks/` | exploratory only, not part of the pipeline |

## The modules at this level

**The frozen prior and its sampling.** `model.py` is the Quetzal architecture, a decoder-only
transformer whose atom-type stack emits logits and whose coordinate stack conditions a
diffusion sampler. `generate.py` runs a rollout, `attention.py` and `simple_mlp.py` are the
building blocks, and `hdeco.py` handles the hierarchical decomposition of a molecule into the
generation order.

**Guides and their training.** `gflow.py` is the main training loop and holds the guide
taxonomy in its module docstring; it defines `LogitGuide`, the plain residual correction to the
logits. `hidden_guide.py` and `tempgain_guide.py` are the other two architectures.
`gflow_multi.py` handles the multi-component case that the composition track builds on, and
`replay_buffer.py` is the reward-prioritised replay used as a swept factor.
`rtb_finetune.py` is the full-weight fine-tuning path, which the dissertation does not use.

**Rewards and chemistry.** `reward_fn.py` assembles the GuacaMol objectives and the
nitrogen-fraction control. `chem.py` converts generated 3D output to molecules,
`edm_metrics.py` computes atom and molecule stability, and `metrics.py` and `density.py` cover
the remaining distributional measures.

**Sampling and aggregation.** `final_dump.py` and `final_dump_composed.py` produce the scored
sample dumps that every result reads. `aggregate_dumps.py` builds the master table.
`harvest_eval.py`, `harvest_analysis.py` and `best_of_n_curve.py` build the best-of-$N$
unguided baseline.

**Data.** `geom.py`, `qm9.py`, `datasets.py`, `data_smiles.py` and `pack.py` load and pack the
training corpora. `make_reference.py` builds the GEOM-Drugs SMILES reference the
corpus-ceiling analysis compares against.

**The discriminator baseline.** `train_fudge.py` and `fudge_data.py` implement the state-only
future discriminator, which uses the same guide architecture under a non-GFlowNet objective.

**Utilities.** `hang_guard.py` watches for stalled runs, `draw.py` and `smiles_hist.py` are
inspection helpers.
