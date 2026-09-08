#!/usr/bin/env bash
# =============================================================================
# Stage 5 -- the composition track.
#
# Mixes the per-component guides from stage 2 into one sampler and scores it on
# the assembled MPO objective the components never saw, under three operators:
#
#   linear    p_M proportional to sum_i W_i          (exact at beta = 1)
#   product   product-of-experts, "all components high"   (approximate)
#   harmonic  the harmonic-mean product operator          (approximate)
#
# where W_i = omega_i * Z_i * u_i(s_t) weights each component by its learned
# partition function and its running probability of having reached the current
# state along the shared trajectory.
#
# One process handles all operators for a given seed (it loops over them
# internally, reusing the loaded checkpoints), so parallelism is over seeds.
#
# Each run emits both compose.py's own reporting -- reward histograms, KDE,
# ternary plot, hypervolume, per-component summary, under <out>/compose_native/
# -- and a dump_summary.json in the same schema as stage 4, so composed rows
# join the master table alongside the single-guide sweep.
#
# Note on routes: --route policy applies the composed residual to the logits.
# An earlier set of runs used a flow-based routing path in which the residual
# was computed but never applied; its rollout diagnostics show a residual norm
# of exactly 0.000 at every state and a flip rate identically zero at every
# position. Those runs are excluded from all reported results as a delivery
# failure rather than a bound.
#
# Usage:
#   bash scripts/05_dump_composed.sh
#   COMPONENTS="0 1 2 3" bash scripts/05_dump_composed.sh   # include the dead axis
#   OPERATORS=harmonic SEEDS=42 bash scripts/05_dump_composed.sh
#   REWARDS="osim peri" GUIDES="hidden base" OBJECTIVES="db rtb" \
#     TRAIN_SEEDS="0 42" BETA=10 SEEDS=0 N=5000 bash scripts/05_dump_composed.sh
#   DRY=1 bash scripts/05_dump_composed.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
warn_missing_ref

OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/dumps_composed}"
SINGLE_ROOT="${SINGLE_ROOT:-${RESULTS_ROOT}/dumps}"
N="${N:-5000}"
SEEDS="${SEEDS:-42}"                # DUMP (sampling) seed(s) -- not the guides' training seed
DIFF_STEPS="${DIFF_STEPS:-18}"
REF_LIMIT="${REF_LIMIT:-$N}"
DATASET="${DATASET:-geom}"
OPERATORS="${OPERATORS:-linear,product,harmonic}"
BETA="${BETA:-10}"
GUIDE_SOURCE="${GUIDE_SOURCE:-ema}"
# generation batch size (final_dump_composed's --chunk); a K=3-4 composed
# sampler at the default chunk=500 was measured peaking near 31GB on a 32GB
# card at N=5000, so this defaults lower to leave headroom for FCD +
# compose_native's own re-sampling pass, both of which run after generation.
CHUNK="${CHUNK:-200}"
MAIN_LOG="${MAIN_LOG:-${LOG_ROOT}/compose_dump_master.log}"
: > "$MAIN_LOG"

# multi-bench / multi-guide-config axes, same short names as stages 1-2.
# Single-bench override (BENCH_KEY/BENCH_TAG) kept for backward compatibility;
# REWARDS takes precedence when set.
BENCH_KEY="${BENCH_KEY:-osimertinib}"
BENCH_TAG="${BENCH_TAG:-osim}"
REWARDS="${REWARDS:-}"
GUIDES="${GUIDES:-hidden}"
OBJECTIVES="${OBJECTIVES:-db}"
TRAIN_SEEDS="${TRAIN_SEEDS:-}"       # empty -> unsuffixed component checkpoints
CONV_PROCS="${CONV_PROCS:-8}"
CONV_TIMEOUT="${CONV_TIMEOUT:-90}"
CONV_ARGS="--conv_procs ${CONV_PROCS} --conv_timeout ${CONV_TIMEOUT}"

# reward -> (bench full standard_benchmarks name, tag, default components).
# Must match scripts/02_train_components.sh's bench_full()/components_for().
bench_full () {
  case "$1" in
    osim) echo "hard_osimertinib" ;;
    peri) echo "perindopril_rings" ;;
    *) echo "" ;;
  esac
}
bench_key_for () {
  case "$1" in
    osim) echo "osimertinib" ;;
    peri) echo "perindopril" ;;
    *) echo "" ;;
  esac
}
components_for () {
  if [[ -n "${COMPONENTS:-}" ]]; then echo "$COMPONENTS"; return; fi
  case "$1" in
    osim) echo "0 1 2 3" ;;
    peri) echo "0 1" ;;
    *) echo "" ;;
  esac
}

if [[ -n "$REWARDS" ]]; then
  REWARD_LIST="$REWARDS"
else
  REWARD_LIST="$BENCH_TAG"
fi

REF_ARG=""
[[ -f "$REF_SMILES" ]] && REF_ARG="--ref_smiles ${REF_SMILES} --ref_limit ${REF_LIMIT}"

RAN=0; START_TS=$(date +%s)

for tag in $REWARD_LIST; do
  if [[ -n "$REWARDS" ]]; then
    full=$(bench_full "$tag"); key=$(bench_key_for "$tag"); comps=$(components_for "$tag")
    if [[ -z "$full" || -z "$key" || -z "$comps" ]]; then
      echo "[fatal] no bench mapping for reward '$tag'; add one to bench_full()/" \
           "bench_key_for()/components_for()" >&2
      exit 1
    fi
  else
    full="hard_${BENCH_KEY}"; key="$BENCH_KEY"; comps="${COMPONENTS:-0 2 3}"
  fi

  for guide in $GUIDES; do
    for obj in $OBJECTIVES; do
      for tseed in ${TRAIN_SEEDS:-""}; do
        CKPTS=""; LABELS=""; WEIGHTS=""; BETAS=""
        NCOMP=$(echo "$comps" | wc -w | tr -d ' ')
        W=$(awk -v n="$NCOMP" 'BEGIN{printf "%.3f", 1.0/n}')
        for c in $comps; do
          cname="compose-${tag}-c${c}-${guide}-${obj}-b${BETA}"
          [[ -n "$tseed" ]] && cname="${cname}-s${tseed}"
          ckpt=$(resolve_ckpt "$cname") || {
            echo "[MISSING] component c${c}: no checkpoint for $cname" | tee -a "$MAIN_LOG"; continue; }
          CKPTS="${CKPTS:+$CKPTS,}${ckpt}"
          LABELS="${LABELS:+$LABELS,}c${c}"
          WEIGHTS="${WEIGHTS:+$WEIGHTS,}${W}"
          BETAS="${BETAS:+$BETAS,}${BETA}"
        done

        if [[ -z "$CKPTS" ]]; then
          echo "[fatal] no component checkpoints for ${tag}/${guide}/${obj}${tseed:+/s$tseed}; " \
               "run scripts/02_train_components.sh first" | tee -a "$MAIN_LOG"
          continue
        fi
        say "composing ${tag} ${guide}/${obj}${tseed:+ s$tseed}: ${NCOMP} components (${LABELS})" \
          | tee -a "$MAIN_LOG"

        # Eval spec: the full MPO objective first (it is the primary metric and
        # the one the aggregator reads), then each component present.
        EVAL="guacamol:${full}=${tag}_MPO"
        for c in $comps; do
          EVAL="${EVAL},gcomp:${key}:${c}=c${c}"
        done

        OUT="${OUT_ROOT}/${key}/${guide}-${obj}${tseed:+-s$tseed}"
        TRAIN_SEED_ARG=""
        [[ -n "$tseed" ]] && TRAIN_SEED_ARG="--train_seed ${tseed}"

        for seed in $SEEDS; do
          done_all=1
          for op in ${OPERATORS//,/ }; do
            [[ -f "${OUT}/${op}/seed${seed}/dump_summary.json" ]] || done_all=0
          done
          if [[ "$done_all" == "1" ]]; then
            echo "[skip] ${tag} ${guide}/${obj}${tseed:+/s$tseed}, all operators, seed$seed" \
              | tee -a "$MAIN_LOG"; continue
          fi

          CMD="$PY final_dump_composed.py \
            --guide_ckpts ${CKPTS} \
            --guide_labels ${LABELS} \
            --weights ${WEIGHTS} \
            --train_betas ${BETAS} \
            --eval_rewards ${EVAL} \
            --bench_key ${key} \
            --guide_arch ${guide} \
            --objective ${obj} \
            ${TRAIN_SEED_ARG} \
            --operators ${OPERATORS} \
            --n ${N} --seed ${seed} --chunk ${CHUNK} \
            --diff_steps ${DIFF_STEPS} --dataset ${DATASET} \
            --guide_source ${GUIDE_SOURCE} --progress ${CONV_ARGS} \
            ${REF_ARG} \
            --out_dir ${OUT}"

          say "compose ${tag} ${guide}/${obj}${tseed:+/s$tseed} dump-seed$seed (operators: ${OPERATORS})" \
            | tee -a "$MAIN_LOG"
          [[ "$DRY" == "1" ]] && { echo "$CMD"; continue; }

          throttle
          GPU=$(( RAN % NUM_GPUS ))
          mkdir -p "$OUT"
          TAG="compose|${tag}|${guide}|${obj}${tseed:+|s$tseed}|s${seed}"
          (
            CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL bash -c "$CMD" 2>&1 \
              | stdbuf -oL tee "${OUT}/compose_seed${seed}.log" \
              | stdbuf -oL tr '\r' '\n' \
              | stdbuf -oL sed "s#^#[${TAG}] #" >> "$MAIN_LOG"
          ) &
          echo "[launch] ${TAG} (pid $!)" | tee -a "$MAIN_LOG"
          RAN=$((RAN+1)); sleep 2
        done
      done
    done
  done
done

wait
END_TS=$(date +%s)
say "composed dumps done: ran=$RAN elapsed $(( (END_TS-START_TS)/60 )) min" | tee -a "$MAIN_LOG"

# Join composed rows into the single-guide master table.
if [[ "$DRY" != "1" ]]; then
  say "aggregating composed + single-guide dumps" | tee -a "$MAIN_LOG"
  $PY aggregate_dumps.py \
    --dumps_root "$SINGLE_ROOT" \
    --extra_roots "$OUT_ROOT" \
    --out_dir "${SINGLE_ROOT}/_aggregate" 2>&1 \
    | stdbuf -oL sed 's#^#[aggregate] #' | tee -a "$MAIN_LOG"
fi
hr
