#!/bin/bash
# Flip diagnostics over every trained guide checkpoint under $CKPT_ROOT.
# Mirrors scripts_molgpt/06_flip_diagnostics.sh -- see that file's comments.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_prior

OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/flips}"
TEMPS="${TEMPS:-1.0}"
N_TRAJ="${N_TRAJ:-500}"
N_REPORT_POS="${N_REPORT_POS:-256}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CKPT_NAME="${CKPT_NAME:-last.ckpt}"

mkdir -p "$OUT_DIR" "$OUT_DIR/logs"

read -r -a TEMP_ARR <<< "$TEMPS"
TEMP_FLAGS="--flip_temp ${TEMP_ARR[0]}"
[[ "${#TEMP_ARR[@]}" -ge 2 ]] && TEMP_FLAGS="$TEMP_FLAGS --also_temp ${TEMP_ARR[1]}"

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

  safe="$(echo "$name" | tr -c '[:alnum:]_.-' '_')"
  json="${OUT_DIR}/flip_report_${safe}.json"
  if [[ "$SKIP_EXISTING" == "1" && -s "$json" ]]; then
    say "[skip] $name (report exists)"
    continue
  fi

  COUNT=$((COUNT+1))
  CMD="$PY g2pt_gfn/flip_g2pt.py --ckpt $ckpt --model_name_or_path $G2PT_MODEL \
    --label $name --report_tag $safe --out_dir $OUT_DIR \
    --n_traj $N_TRAJ --n_report_pos $N_REPORT_POS $TEMP_FLAGS"

  [[ "$DRY" == "1" ]] && { hr; echo "[$COUNT] $name"; echo "$CMD"; continue; }

  say "[$COUNT] $name"
  if eval "$CMD" > "${OUT_DIR}/logs/${safe}.log" 2>&1; then
    say "[done] $name"
  else
    say "[FAIL] $name (see ${OUT_DIR}/logs/${safe}.log)"
    tail -20 "${OUT_DIR}/logs/${safe}.log"
  fi
done

echo "processed $COUNT run(s)"
