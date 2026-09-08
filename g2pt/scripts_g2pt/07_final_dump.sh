#!/bin/bash
# Final N=5000 evaluation dump over every trained guide checkpoint, matching
# final_dump.py's convention (see final_dump_g2pt.py's module docstring).
# Same skip/resume discipline as 06_flip_diagnostics.sh.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_prior

OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/dumps}"
N="${N:-5000}"
GUIDE_SOURCE="${GUIDE_SOURCE:-ema}"
# 128, not molgpt's 256: matches the bsz already confirmed safe at
# max_len=300 on an uncontended GPU (see vendor/PROVENANCE.md) -- generate_guided's
# memory profile here is the same rollout machinery training uses.
CHUNK="${CHUNK:-128}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CKPT_NAME="${CKPT_NAME:-last.ckpt}"
# SAMPLE_TEMP (added 2026-09-06, author instruction): empty means "use the
# checkpoint's own cfg.sample_temp" (final_dump_g2pt.py's original default,
# 2.0 for every config here since 01_train_guides.sh never overrides it at
# train time). Author instruction, same day: evaluate this leg's guides at
# sample_temp=1.0 instead -- at 2.0, G2PT's own frozen prior's validity
# collapses to ~1% (vs ~24.5% at 1.0, see decisions.md), which floors
# guided and base parse rate to the same noise level and makes every
# validity/landscape/chemspace figure for this leg uninformative regardless
# of what the guide does. This does NOT touch flip diagnostics -- those
# already evaluate at flip_temp=1.0 by their own separate default (see
# flip_g2pt.py), so delivered_frac/argmax_flip_rate are unaffected either
# way. Quetzal's and MolGPT's own final_dump conventions are unchanged
# (both still record sample_temp=2.0, confirmed from their real dump
# summaries) -- this is now a deliberate, disclosed difference in eval
# temperature for this one leg, not an attempt at cross-model parity.
SAMPLE_TEMP_FLAG=""
[[ -n "${SAMPLE_TEMP:-}" ]] && SAMPLE_TEMP_FLAG="--sample_temp ${SAMPLE_TEMP}"

mkdir -p "$OUT_DIR" "$OUT_DIR/logs"

COUNT=0
for run_dir in "${CKPT_ROOT}"/sweep-*; do
  [[ -d "$run_dir" ]] || continue
  name="$(basename "$run_dir")"
  ckpt="${run_dir}/checkpoints/${CKPT_NAME}"
  if [[ ! -f "$ckpt" ]]; then
    ckpt="$(ls -t "${run_dir}/checkpoints"/*.ckpt 2>/dev/null | head -1)"
  fi
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    say "[skip] $name (no checkpoint found)"
    continue
  fi

  run_out="${OUT_DIR}/${name}"
  if [[ "$SKIP_EXISTING" == "1" && -s "${run_out}/summary.json" ]]; then
    say "[skip] $name (summary exists)"
    continue
  fi

  COUNT=$((COUNT+1))
  mkdir -p "$run_out"
  CMD="$PY g2pt_gfn/final_dump_g2pt.py --ckpt $ckpt --n $N \
    --guide_source $GUIDE_SOURCE --chunk $CHUNK --out_dir $run_out $SAMPLE_TEMP_FLAG"

  [[ "$DRY" == "1" ]] && { hr; echo "[$COUNT] $name"; echo "$CMD"; continue; }

  say "[$COUNT] $name"
  throttle
  ( GPU="$(acquire_gpu)"
    trap 'release_gpu "$GPU"' EXIT
    export CUDA_VISIBLE_DEVICES="$GPU"
    if eval "$CMD" > "${OUT_DIR}/logs/${name}.log" 2>&1; then
      say "[done] $name (GPU $GPU)"
    else
      say "[FAIL] $name (GPU $GPU, see ${OUT_DIR}/logs/${name}.log)"
      tail -20 "${OUT_DIR}/logs/${name}.log"
    fi
  ) &
  sleep 2
done

wait
echo "processed $COUNT run(s)"
