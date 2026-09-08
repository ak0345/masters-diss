#!/usr/bin/env bash
# Perindopril composition-weight sweep: omega_1 from 0 to 1 in steps of 0.1, with
# omega_2 = 1 - omega_1, at every mixing operator. Perindopril has exactly two live
# components, so the simplex is a line and 11 points cover it completely.
#
# The question is whether uniform weighting caps the composed sampler. If it does,
# some interior weighting should reach assembled scores the uniform point cannot.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/common.sh"

CK="${CKPT_ROOT:-logs/quetzal-gfn}"
C0="${REPO_ROOT}/${CK}/compose-peri-c0-hidden-db-b10-s0/checkpoints/last.ckpt"
C1="${REPO_ROOT}/${CK}/compose-peri-c1-hidden-db-b10-s0/checkpoints/last.ckpt"
for f in "$C0" "$C1"; do [[ -f "$f" ]] || { echo "[fatal] missing $f"; exit 1; }; done
OUT_ROOT="${OUT_ROOT:-results/ablations/peri-weights}"
N="${N:-800}"; CHUNK="${CHUNK:-200}"; OPERATORS="${OPERATORS:-linear,product,harmonic}"
EVAL="guacamol:perindopril_rings=peri_MPO,gcomp:perindopril:0=c0,gcomp:perindopril:1=c1"
mkdir -p "$OUT_ROOT"

i=0
for w0 in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  w1=$(awk -v a="$w0" 'BEGIN{printf "%.1f", 1.0-a}')
  OUT="${OUT_ROOT}/w${w0}"
  if [[ -f "${OUT}/harmonic/seed0/dump_summary.json" ]]; then
    echo "[skip] w0=$w0 already done"; continue
  fi
  NG="${NUM_GPUS:-1}"
  GPU=$(( i % NG )); i=$((i+1))
  echo "[run] omega=($w0,$w1) on GPU $GPU"
  (
    CUDA_VISIBLE_DEVICES=$GPU python final_dump_composed.py \
      --guide_ckpts "${C0},${C1}" --guide_labels c0,c1 \
      --weights "${w0},${w1}" --train_betas 10,10 \
      --eval_rewards "$EVAL" --bench_key perindopril \
      --guide_arch hidden --objective db --train_seed 0 \
      --operators "$OPERATORS" --n "$N" --seed 0 --chunk "$CHUNK" \
      --diff_steps 18 --dataset geom --guide_source ema \
      --out_dir "$OUT" > "${OUT_ROOT}/w${w0}.log" 2>&1 \
      || echo "[fail] w0=$w0 (see ${OUT_ROOT}/w${w0}.log)"
  ) &
  # One job per visible GPU, then drain before filling the next batch.
  if (( i % NG == 0 )); then wait; fi
done
wait
echo "sweep complete"
