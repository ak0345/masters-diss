# molgpt-gfn

The MolGPT leg of `../plan_molgpt.md`: same guide machinery as the main
`steerability-limits` paper (frozen prior, hidden-state/logit guides, DB/RTB
training, flip diagnostics), pointed at MolGPT (a SMILES-token GPT,
`devalab/molgpt`) instead of Quetzal (a 3D geometric generator), to test
whether the paper's central finding is a property of Quetzal's geometric
conditioning or of logit-level guidance on a frozen autoregressive prior more
generally. Read `../plan_molgpt.md` first — this README tracks status against
that plan's own sequencing, it doesn't restate the reasoning behind it.

This directory is self-contained and git-ignored from the parent
`quetzal_gfn` repo (see the parent `.gitignore`) — it's meant to become its
own GitHub repository, not to live inside this one long-term.

## Status (2026-09-03)

**Environment ready, retrain-from-scratch in progress, code scaffold not yet
run end-to-end.** `conda env update` has been run (torch 2.7.1+cu128,
CUDA available, rdkit, guacamol all present). GPU is 3x RTX PRO 4500
Blackwell (32GB each, 96GB total).

### The pretrained checkpoint turned out to be a dead end — two real findings, not just missing credentials

1. **Kaggle's `guacamol_nocond.pt` is a mislabeled duplicate of
   `guacamol_qed.pt`**, not the unconditional checkpoint its name claims.
   Confirmed by diffing full state dicts (byte-identical) after first hitting
   a `prop_nn`/mask-shape load error diagnosing as a real `num_props=1`
   architecture, not the `num_props=0` one `train_guacamol.sh` documents.
   Behaviourally confirmed too: feeding it different constant "property"
   values swung validity from 62.5% (constant=0.0) to 95%+ (constant≥0.5) —
   a real conditioning signal, not an inert layer. There is no genuinely
   unconditional checkpoint in that Kaggle dataset. Full detail in
   `vendor/PROVENANCE.md`.
2. **The vendor README's own `guacamol2.csv` Google Drive link is also
   dead** (empty folder, checked directly — not a permissions wall). Rebuilt
   `vendor/molgpt/datasets/guacamol2.csv` from GuacaMol's official figshare
   release instead (md5-verified against the checksums the `guacamol`
   package's own README publishes) — see `vendor/PROVENANCE.md` for exactly
   how.

Both are now resolved by going straight to `plan_molgpt.md`'s documented
fallback: retrain MolGPT from scratch (`train_guacamol.sh`'s first,
`num_props=0` line) rather than trust either external artifact further.

Every module under `molgpt_gfn/` was written against the vendored MolGPT
source and the parent repo's confirmed reuse interfaces (loss math and guide
classes read and matched line-for-line, not paraphrased — see
`vendor/PROVENANCE.md`), but **nothing there has executed end-to-end yet**.
Don't trust a number out of this pipeline until sequencing step 4 below (flip
diagnostic on one config, delivery ≈ 1.0) has actually passed.

### Running it all: `scripts_molgpt/run_pipeline.sh`

Chains retrain → sanity check (hard-gated: refuses to proceed if
`valid_frac < 0.2`) → the 24-run guide sweep → flip diagnostics, stopping at
the first stage that fails rather than burning GPU time on a broken prior.
GPU work is round-robin pinned across however many GPUs `nvidia-smi` reports
(`NUM_GPUS`, overridable), one run per card by default
(`MAX_PARALLEL=$NUM_GPUS`).

```sh
cd molgpt
nohup bash scripts_molgpt/run_pipeline.sh > logs/drivers/pipeline.log 2>&1 &
disown
tail -f logs/drivers/pipeline.log       # or logs/drivers/molgpt_pretrain.log during stage 1
```

`nohup ... & disown` detaches it from this session entirely — it keeps
running and is monitorable with plain `tail`/`ps`/`nvidia-smi` from any
terminal on this machine, independent of whether this Claude session is
still open. Each stage also has its own log under `logs/drivers/` (pretrain,
sanity, and per-run guide-training logs under `logs/drivers/guides/`, flip
diagnostics under `results/molgpt/flips/logs/`).

### What's confirmed (by reading source, not by running it, except where noted)

See `vendor/PROVENANCE.md` for the detailed version. Headline items: the LM
head is a plain `nn.Linear` (no tied embeddings) applied post-LayerNorm, so
`HiddenGuide`/`LogitGuide` port unmodified; there's no KV-cache; the
`guacamol2.csv` reconstruction's tokenization matches `train.py`'s hardcoded
94-token vocab exactly (verified against `guacamol2_stoi.json`, not assumed).

### What's next, in order (mirrors plan_molgpt.md's "Sequencing")

1. ~~Resolve the two blockers~~ — done (env installed; checkpoint path
   switched to retrain-from-scratch after the Kaggle/Drive dead ends above).
2. Retrain (`run_pipeline.sh` stage 1) + sanity check (stage 2) — in
   progress / automated by the pipeline script.
3. One config end-to-end, if not already covered by the pipeline's own
   stage-2 gate: `python molgpt_gfn/gflow_molgpt.py --name smoke --reward
   nitrogen_count --objective db --max_epochs 1 --steps_per_epoch 20 --bsz
   32`, then `python molgpt_gfn/flip_molgpt.py --ckpt
   logs/molgpt-gfn/smoke/checkpoints/last.ckpt --label smoke --n_traj 200`.
   If `delivered_frac` isn't ≈1.0 immediately, that's a wiring bug.
4. `DRY=1 bash scripts_molgpt/01_train_guides.sh` to preview the 24-config
   grid, or just let `run_pipeline.sh` run it (stage 3).
5. `bash scripts_molgpt/06_flip_diagnostics.sh` (stage 4 of the pipeline).
6. Aggregate into `results/molgpt/master_table.csv` — column semantics
   matched to `../results/dumps/_aggregate/master_table.csv` where they
   overlap (not merged into it — separate model, kept distinguishable per
   the plan). **Not yet written** — no runs exist yet to aggregate.
7. Composed guides, only if time and diagnosis budget remain — see
   plan_molgpt.md's "Composed guides" section before touching this; it is
   explicitly not free (the 3D paper's own composed-guide runs hit a silent
   no-op bug and were dropped from the manuscript over it).

## Layout

```
molgpt_gfn/
  molgpt_prior.py   frozen MolGPT wrapper: load, freeze, expose per-step hidden states
  guides.py         HiddenGuide / LogitGuide / LogFlowHead, ported from hidden_guide.py / gflow.py
  config.py         GFNConfig, adapted from gflow.py's (coordinate/mask fields dropped)
  reward_adapter.py SMILES-native guacamol + nitrogen-fraction reward, ported from reward_fn.py
  gflow_molgpt.py   training loop: rollout builders + DB/RTB loss (loss math ported verbatim)
scripts_molgpt/
  common.sh, 01_train_guides.sh, 06_flip_diagnostics.sh   mirror scripts/'s interface and conventions
vendor/molgpt/      devalab/molgpt, vendored read-only (see vendor/PROVENANCE.md)
checkpoints/        pretrained + trained-guide checkpoints (git-ignored)
logs/molgpt-gfn/    run directories (git-ignored)
results/molgpt/     flip reports, master_table.csv (git-ignored except the aggregate table)
```

## Deliberate simplifications relative to the parent repo's `gflow.py`

Documented in full in `gflow_molgpt.py`'s module docstring; summary:

- **No PyTorch Lightning** — a plain training loop. A MolGPT guide run is
  small (plan estimate: under 30 minutes on one A100); Lightning's
  multi-GPU/callback machinery buys nothing here and adds a dependency and
  config surface nothing exercises.
- **No guide EMA.** Same "small run" reasoning. Worth adding if results turn
  out sensitive to it — don't add it speculatively.
- **No hang-guard.** `reward_adapter.py`'s reward path is `Chem.MolFromSmiles`
  + a GuacaMol scorer, no geometry step — much less hang-prone than Quetzal's
  `rdDetermineBonds`/xtb path the hang guard exists to catch. That's an
  assumption, not a proof; if a sweep run hangs, add a guard rather than
  trusting this note.
- **Replay and warm-start both refuse loudly** (`NotImplementedError`) rather
  than silently no-op — out of scope per the pilot's grid (`plan_molgpt.md`:
  "Replay: off only"), and `replay_buffer.py`'s storage format is
  Quetzal-shaped (atoms+coords), not something to adapt speculatively.

## Checkpoint format

`gflow_molgpt.py` writes a plain `torch.save` dict (`config`,
`guide_state_dict`, `guide_class`, `flow_head_state_dict` if DB, `logZ` if
RTB, `step`, `epoch`) — **not** a Lightning `.ckpt`. `flip_molgpt.py`'s
`load_guide_checkpoint` reads this format and, matching the parent repo's
`final_dump.py` discipline, raises immediately on a guide-type mismatch or
missing guide weights rather than silently scoring an untrained guide.
