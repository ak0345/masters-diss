#!/bin/bash
# Full unattended pipeline: retrain MolGPT (unconditional, guacamol) -> sanity
# check -> 48-run guide sweep (with EMA) -> flip diagnostics. Meant to run under nohup;
# see ../README.md, "How to run it all", for how to launch and monitor this.
#
# Stops at the first stage that fails rather than burning GPU time on top of
# a broken prior/guide -- each stage's exit code is checked explicitly, not
# just left to `set -e` (which a background `| tee` pipeline would silently
# swallow via PIPESTATUS if not handled).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

VENDOR_DIR="${REPO_ROOT}/vendor/molgpt"
PRETRAIN_SRC="${REPO_ROOT}/vendor/cond_gpt/weights/geom_drugs_unconditional.pt"
mkdir -p "$LOG_ROOT"

say "=== Pipeline start: $NUM_GPUS GPU(s) detected, MAX_PARALLEL=$MAX_PARALLEL ==="

# ---------------------------------------------------------------- Stage 1
say "=== Stage 1/6: retrain MolGPT from scratch on GEOM-Drugs (unconditional, num_props=0) ==="
if [[ -f "$MOLGPT_CKPT" ]]; then
  say "checkpoint already exists at $MOLGPT_CKPT -- skipping retrain"
else
  if [[ ! -f "${VENDOR_DIR}/datasets/geom_drugs.csv" ]]; then
    say "[FATAL] ${VENDOR_DIR}/datasets/geom_drugs.csv not found -- build it first"
    exit 1
  fi
  (
    cd "$VENDOR_DIR" || exit 1
    # vendor/molgpt's train/utils.py and train/trainer.py both import
    # moses.utils unconditionally though this run (generate=False) never
    # calls it -- see vendor/moses_stub/moses/utils.py for why a stub
    # instead of the real (broken-to-build-here) `moses` package.
    export PYTHONPATH="${REPO_ROOT}/vendor/moses_stub${PYTHONPATH:+:${PYTHONPATH}}"
    GPU="$(acquire_gpu)"
    trap 'release_gpu "$GPU"' EXIT
    # Author instruction, 2026-09-04: wandb online (already logged in),
    # not offline -- this run is long enough to be worth monitoring live.
    WANDB_MODE=online CUDA_VISIBLE_DEVICES="$GPU" "$PY" train/train.py \
      --run_name geom_drugs_unconditional --data_name geom_drugs \
      --batch_size 384 --num_props 0 --max_epochs 10
  ) > "${LOG_ROOT}/molgpt_pretrain.log" 2>&1
  PRETRAIN_STATUS=$?
  # Don't gate on exit code alone: vendor/molgpt/train/train.py's own last
  # line (`df.to_csv(...)` on a `df` that's None whenever `generate=False`,
  # the vendor's own default training mode) crashes AFTER a fully successful
  # training run and checkpoint save -- confirmed 2026-09-03, see
  # vendor/PROVENANCE.md. Checkpoint existence is the real signal.
  if [[ $PRETRAIN_STATUS -ne 0 ]]; then
    say "pretraining exited $PRETRAIN_STATUS -- checking whether that's the known"
    say "harmless trailing df.to_csv(None) crash (see vendor/PROVENANCE.md) or a real failure"
  fi
  if [[ ! -f "$PRETRAIN_SRC" ]]; then
    say "[FATAL] no checkpoint at $PRETRAIN_SRC -- pretraining genuinely failed, see ${LOG_ROOT}/molgpt_pretrain.log"
    exit 1
  fi
  mkdir -p "$(dirname "$MOLGPT_CKPT")"
  cp -v "$PRETRAIN_SRC" "$MOLGPT_CKPT"
  cp -v "${REPO_ROOT}/vendor/cond_gpt/weights/geom_drugs_unconditional_vocab_manifest.json" \
        "${REPO_ROOT}/vendor/molgpt/geom_drugs_vocab_manifest.json"
  say "pretraining done, checkpoint copied to $MOLGPT_CKPT"
fi

# ---------------------------------------------------------------- Stage 2
say "=== Stage 2/6: sanity check ==="
# CUDA_VISIBLE_DEVICES=$GPU_OFFSET (2026-09-04): see g2pt's run_pipeline.sh
# for why -- keeps this one-shot check inside this leg's GPU partition
# instead of always landing on GPU0 by default.
CUDA_VISIBLE_DEVICES="$GPU_OFFSET" "$PY" molgpt_gfn/molgpt_prior.py "$MOLGPT_CKPT" --batch_size 64 --device cuda \
  --min_valid_frac 0.2 > "${LOG_ROOT}/molgpt_sanity.log" 2>&1
SANITY_STATUS=$?
cat "${LOG_ROOT}/molgpt_sanity.log"
if [[ $SANITY_STATUS -ne 0 ]]; then
  say "[FATAL] sanity check failed (see ${LOG_ROOT}/molgpt_sanity.log) -- refusing to start the sweep"
  exit 1
fi
say "sanity check passed"

# Author instruction, 2026-09-04: "the pipeline for each is just the
# retraining then guide training then eval and then flip diagnostics" --
# final_dump/aggregate (eval) run BEFORE flip diagnostics, not after.

# ---------------------------------------------------------------- Stage 3
# Exit-code checks added 2026-09-05 (see g2pt's run_pipeline.sh for the
# incident that motivated this): a hard failure in any stage here used to
# let the script cascade through every remaining one and print "Pipeline
# complete" regardless. Halting surfaces that immediately instead.
say "=== Stage 3/6: guide sweep (with EMA, across $NUM_GPUS GPU(s)) ==="
if ! bash "${SCRIPT_DIR}/01_train_guides.sh"; then
  say "[FATAL] guide sweep exited non-zero -- refusing to run eval/flip on whatever partial state that left"
  exit 1
fi
say "guide sweep finished"

# ---------------------------------------------------------------- Stage 4
say "=== Stage 4/6: final N=5000 eval dump (guide_source=ema) ==="
if ! bash "${SCRIPT_DIR}/07_final_dump.sh"; then
  say "[FATAL] final dump exited non-zero"
  exit 1
fi
say "final dump finished"

# ---------------------------------------------------------------- Stage 5
say "=== Stage 5/6: aggregate into master_table.csv ==="
if ! bash "${SCRIPT_DIR}/08_aggregate.sh"; then
  say "[FATAL] aggregate exited non-zero"
  exit 1
fi
say "aggregate finished"

# ---------------------------------------------------------------- Stage 6
say "=== Stage 6/6: flip diagnostics (guide_source=ema) ==="
if ! bash "${SCRIPT_DIR}/06_flip_diagnostics.sh"; then
  say "[FATAL] flip diagnostics exited non-zero"
  exit 1
fi
say "flip diagnostics finished"

say "=== Pipeline complete ==="
