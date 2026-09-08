# scripts_molgpt/

This leg's pipeline. Same conventions as the Quetzal `scripts/`: `DRY=1` prints commands without
running them, and stages skip work whose output already exists.

Source `common.sh` or invoke through these drivers, never by calling the Python entry points from
another directory: `common.sh` sets `REPO_ROOT` and `PYTHONPATH`, and the imports fail without it.

| script | stage |
|---|---|
| `run_pipeline.sh` | everything unattended: retrain the prior, sanity check, guide sweep, diagnostics |
| `01_train_guides.sh` | the guide sweep |
| `06_flip_diagnostics.sh` | flip diagnostics over the trained guides |
| `07_final_dump.sh` | sample and score each checkpoint |
| `08_aggregate.sh` | pool the dumps into this leg's master table |
| `common.sh` | shared paths, GPU detection, parallelism, logging |

The stage numbers match the Quetzal pipeline's, which is why they are not contiguous. Stages
2 to 5 and 9 have no equivalent here: this leg runs no composition track, no fine-tuning and no
cluster-occupancy analysis.

`run_pipeline.sh` stops at the first stage that fails rather than burning GPU time on top of a
broken prior. It checks each stage's exit code explicitly, because a backgrounded `| tee`
pipeline would otherwise swallow the failure.

## Sweep scope

Deliberately narrower than Quetzal's: the hidden guide only, DB and RTB, `beta=10`, replay off,
three seeds (0, 42, 100). The purpose is to test whether Chapter 3's finding survives a change of
frozen model, not to re-run the whole design space on a second and third architecture.
