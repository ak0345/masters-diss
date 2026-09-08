#!/bin/bash
# Full unattended pipeline for the G2PT leg: retrain from scratch on
# GEOM-Drugs -> sanity check -> 48-run guide sweep (with EMA) -> final dump
# -> aggregate -> flip diagnostics. Mirrors scripts_molgpt/run_pipeline.sh.
# Stage 1 retrain added 2026-09-04: the old pretrained GuacaMol checkpoint
# (xchen16/g2pt-guacamol-small-deg) is retired -- author instruction, both
# new legs train on GEOM-Drugs with Quetzal's own atom vocabulary, and no HF
# checkpoint exists for that combination. Meant to run under nohup.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
mkdir -p "$LOG_ROOT"

VENDOR_DIR="${REPO_ROOT}/vendor/G2PT"
CKPT_PT="${VENDOR_DIR}/results/geom_drugs-small-bfs/ckpt.pt"

say "=== Pipeline start: $NUM_GPUS GPU(s) detected, MAX_PARALLEL=$MAX_PARALLEL ==="

# ---------------------------------------------------------------- Stage 1
say "=== Stage 1/6: retrain G2PT from scratch on GEOM-Drugs (small/bfs) ==="
if [[ -f "${G2PT_MODEL}/config.json" ]]; then
  say "HF-converted checkpoint already exists at $G2PT_MODEL -- skipping retrain"
else
  if [[ ! -f "${VENDOR_DIR}/datasets/geom_drugs/data_meta.json" ]]; then
    say "[FATAL] ${VENDOR_DIR}/datasets/geom_drugs/ not found -- run datasets/prepare_geom_drugs.py first"
    exit 1
  fi
  (
    cd "$VENDOR_DIR" || exit 1
    GPU="$(acquire_gpu)"
    trap 'release_gpu "$GPU"' EXIT
    # ~20 epochs (437 iters/epoch at effective batch 480, block_size=700):
    # smoke-tested at ~300ms/iter on an uncontended GPU (2026-09-04), so
    # ~44 minutes wall-clock -- cheap enough not to skimp on epochs.
    # always_save_checkpoint=True: matches MolGPT's retrain always saving
    # every epoch, not gated on val loss improving.
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
      configs/networks/small.py configs/datasets/geom_drugs.py configs/default.py \
      --max_iters=8740 --warmup_iters=200 --lr_decay_iters=8740 \
      --eval_interval=437 --eval_iters=50 --log_interval=50 \
      --always_save_checkpoint=True --compile=False
  ) > "${LOG_ROOT}/g2pt_pretrain.log" 2>&1
  PRETRAIN_STATUS=$?
  if [[ $PRETRAIN_STATUS -ne 0 ]]; then
    say "[FATAL] G2PT pretraining exited $PRETRAIN_STATUS -- see ${LOG_ROOT}/g2pt_pretrain.log"
    exit 1
  fi
  if [[ ! -f "$CKPT_PT" ]]; then
    say "[FATAL] no checkpoint at $CKPT_PT -- pretraining didn't save one, see ${LOG_ROOT}/g2pt_pretrain.log"
    exit 1
  fi
  say "pretraining done, converting to HF format..."
  (
    cd "$VENDOR_DIR" || exit 1
    "$PY" convert_to_hf.py "$CKPT_PT" "results/geom_drugs-small-bfs/hf"
  ) > "${LOG_ROOT}/g2pt_convert_to_hf.log" 2>&1
  if [[ ! -f "${VENDOR_DIR}/results/geom_drugs-small-bfs/hf/config.json" ]]; then
    say "[FATAL] HF conversion failed -- see ${LOG_ROOT}/g2pt_convert_to_hf.log"
    exit 1
  fi
  mkdir -p "$(dirname "$G2PT_MODEL")"
  rm -rf "$G2PT_MODEL"
  cp -rv "${VENDOR_DIR}/results/geom_drugs-small-bfs/hf" "$G2PT_MODEL"
  say "conversion done, HF model copied to $G2PT_MODEL"
fi

say "=== Stage 2/6: sanity check ==="
# CUDA_VISIBLE_DEVICES=$GPU_OFFSET (2026-09-04): this doesn't go through
# acquire_gpu (it's a one-shot check, not worth a lock), but it must still
# stay inside this leg's GPU_OFFSET/NUM_GPUS partition -- left on the
# default device it always lands on GPU0 regardless of partition, which
# collides with the other leg's exclusive GPU0 job when both run concurrently.
CUDA_VISIBLE_DEVICES="$GPU_OFFSET" "$PY" g2pt_gfn/g2pt_prior.py --model_name_or_path "$G2PT_MODEL" --batch_size 64 --device cuda \
  --min_valid_frac 0.2 > "${LOG_ROOT}/g2pt_sanity.log" 2>&1
SANITY_STATUS=$?
cat "${LOG_ROOT}/g2pt_sanity.log"
if [[ $SANITY_STATUS -ne 0 ]]; then
  say "[FATAL] sanity check failed (see ${LOG_ROOT}/g2pt_sanity.log) -- refusing to start the sweep"
  exit 1
fi
say "sanity check passed"

# Author instruction, 2026-09-04: "the pipeline for each is just the
# retraining then guide training then eval and then flip diagnostics" --
# final_dump/aggregate (eval) run BEFORE flip diagnostics, not after.

# max_len's real default lives in 01_train_guides.sh (currently 300) -- this
# echo just mirrors that fallback, don't rely on it as the source of truth.
say "=== Stage 3/6: guide sweep (with EMA, across $NUM_GPUS GPU(s), max_len=${MAX_LEN:-300}) ==="
# Exit-code checks added 2026-09-05: this used to call each remaining stage
# unconditionally, so a hard failure early (e.g. 01_train_guides.sh's own
# require_prior exiting 1) still let the script cascade through eval/
# aggregate/flip and print "Pipeline complete" with zero trained configs --
# confirmed happening live (require_prior's batch_size=16 sanity draw hit
# ordinary sampling noise, exited 1, and the pipeline raced through every
# remaining stage in under a minute). Halting here instead surfaces that
# immediately instead of a fully green-looking log with no real output.
if ! bash "${SCRIPT_DIR}/01_train_guides.sh"; then
  say "[FATAL] guide sweep exited non-zero -- refusing to run eval/flip on whatever partial state that left"
  exit 1
fi
say "guide sweep finished"

say "=== Stage 4/6: final N=5000 eval dump (guide_source=ema) ==="
if ! bash "${SCRIPT_DIR}/07_final_dump.sh"; then
  say "[FATAL] final dump exited non-zero"
  exit 1
fi
say "final dump finished"

say "=== Stage 5/6: aggregate into master_table.csv ==="
if ! bash "${SCRIPT_DIR}/08_aggregate.sh"; then
  say "[FATAL] aggregate exited non-zero"
  exit 1
fi
say "aggregate finished"

say "=== Stage 6/6: flip diagnostics (guide_source=ema) ==="
if ! bash "${SCRIPT_DIR}/06_flip_diagnostics.sh"; then
  say "[FATAL] flip diagnostics exited non-zero"
  exit 1
fi
say "flip diagnostics finished"

say "=== Pipeline complete ==="
