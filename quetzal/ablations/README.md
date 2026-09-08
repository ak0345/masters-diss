# ablations/

Mechanism probes. These answer *how* a guide changed generation, which a terminal score cannot.
Stage 7 of the pipeline drives most of them; see `../scripts/07_ablations.sh`.

Unlike the training stages, nothing here is resumable. Each script re-runs every time.

## The flip diagnostics

`single_flip_ablation.py` is the core probe. It rolls trajectories from the frozen prior and,
at every visited state, records four things: whether the guide changed the logits at all
(delivery), how far the atom distribution moved (total variation and KL), whether a
shared-uniform coupling selects a different atom (sample flip), and the prior's own top-1 logit
margin, which is the axis the other three are reported against.

`single_flip_agg.py` and `aggregate_flips.py` pool those per-run reports into the tables the
manuscript uses. `ablate_logit_flip_compose.py` runs the same probe over a composed sampler
rather than a single guide.

**Group flip reports by the guide tag in the run name, never by the `guide_type` field.** The
harness instantiates the residual guide through the `TempGainGuide` class, because the two
coincide at `T = 1`, so `guide_type` silently merges two separately trained populations.

## Where the reward lives

`diag_atoms_vs_coords.py` answers whether the objective is reachable through atom type at all.
It forces every atom in a sampled state to the guide's own top-1 choice with coordinates held
fixed, then separately re-rolls the coordinate diffusion with atoms held fixed, and compares
the two effect sizes.

## Ceilings and capacity

`ablate_ceiling.py` bins flip rate by the prior's confidence margin and measures the variance
each leaf scorer of an MPO objective carries, which is what shows a component with no gradient.
`ablate_guide.py` runs the residual scale sweep, multiplying the trained residual by a constant
at sampling time. `ablate_hidden_guide.py` and `probe_tempgain.py` are the per-architecture
equivalents. `ablate_singles_weights.py` sweeps the composition weight vector.

## Diagnostics on the runs themselves

`diag_training_logs.py` reads local wandb datastores to recover training-batch validity per run,
which is what tests whether low training signal explains a null result.
`diag_rollout.py` inspects a single rollout step by step.
`exclude_decoder_artifacts.py` filters molecules carrying a formal charge on carbon, the
signature of a 3D-to-SMILES bond-perception failure, so the headline numbers can be recomputed
without them. `cluster_occupancy.py` backs the supplementary occupancy analysis.
