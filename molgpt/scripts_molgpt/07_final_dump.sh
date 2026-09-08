#!/bin/bash
# Final N=5000 evaluation dump over every trained guide checkpoint, matching
# final_dump.py's convention (see final_dump_molgpt.py's module docstring).
# Same skip/resume discipline as 06_flip_diagnostics.sh.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_prior

OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/dumps}"
N="${N:-5000}"
GUIDE_SOURCE="${GUIDE_SOURCE:-ema}"
CHUNK="${CHUNK:-256}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CKPT_NAME="${CKPT_NAME:-last.ckpt}"

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
  CMD="$PY molgpt_gfn/final_dump_molgpt.py --ckpt $ckpt --n $N \
    --guide_source $GUIDE_SOURCE --chunk $CHUNK --out_dir $run_out"

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
