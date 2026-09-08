#!/usr/bin/env bash
# =============================================================================
# Stage 2 -- per-component guides, and the atom-stability control.
#
# Two families of guide that sit outside the main sweep of stage 1:
#
#   components  One guide per leaf scorer of an assembled MPO objective, via
#               --reward guacamol_component --reward_component <i>. These are
#               the teachers the composition track mixes in stage 5. Osimertinib
#               has four (0..3); component 1 (ECFP6 similarity) has zero variance
#               over reachable molecules, so its guide receives no gradient and
#               its curve is flat by construction, but it trains like any other
#               -- COMPONENTS defaults to every leaf index of the benchmark
#               rather than pre-excluding it. Perindopril has two (0..1).
#
#               Same axes as stage 1's guide sweep -- REWARDS (bench), GUIDES
#               (hidden/base), OBJECTIVES (db/rtb), REPLAYS, BETAS, SEEDS --
#               so a composed configuration can be trained at whatever point
#               in that grid the composition track needs. SEEDS is opt-in
#               exactly as in stage 1: empty trains one unsuffixed run per
#               configuration; set it to train one run per (configuration,
#               seed) with a -s<N> suffix.
#
#   stability   Guides trained against EDM atom stability rather than a GuacaMol
#               objective. Dense and atom-type-driven, so it is the axis where a
#               logit-level guide should work if anything does. Reported as an
#               excluded run: molecular stability sits below 0.05 under the prior
#               for 80-100 heavy-atom molecules, so the objective is close to
#               saturated and carries little gradient.
#
# Unlike stage 1 this runs with eval_base on for the stability family, since the
# guided-minus-prior delta is the whole question there.
#
# Usage:
#   bash scripts/02_train_components.sh              # both families
#   bash scripts/02_train_components.sh components
#   bash scripts/02_train_components.sh stability
#   COMPONENTS="0 1 2 3" bash scripts/02_train_components.sh components
#   REWARDS="osim peri" GUIDES="hidden base" OBJECTIVES="db rtb" \
#     REPLAYS=off BETAS=10 SEEDS="0 42" bash scripts/02_train_components.sh components
#   DRY=1 bash scripts/02_train_components.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_prior

WHICH="${1:-all}"

# Single-bench override, kept for backward compatibility with the original
# one-benchmark invocation; REWARDS below is the multi-bench axis and takes
# precedence when set.
BENCH="${BENCH:-hard_osimertinib}"
BENCH_TAG="${BENCH_TAG:-osim}"

REWARDS="${REWARDS:-}"          # empty -> fall back to BENCH/BENCH_TAG above
GUIDES="${GUIDES:-hidden}"
OBJECTIVES="${OBJECTIVES:-db}"
REPLAYS="${REPLAYS:-off}"
BETAS="${BETAS:-${BETA:-10}}"
SEEDS="${SEEDS:-}"              # empty -> one unsuffixed run per configuration
MAX_EPOCHS="${MAX_EPOCHS:-5}"
STEPS="${STEPS:-100}"
LOGDIR="${LOGDIR:-${LOG_ROOT}/components}"
mkdir -p "$LOGDIR"

# reward -> (bench full name, components) -- same short names as stage 1.
# COMPONENTS, if set, overrides the per-bench default for every reward.
bench_full () {
  case "$1" in
    osim) echo "hard_osimertinib" ;;
    peri) echo "perindopril_rings" ;;
    *) echo "" ;;
  esac
}
components_for () {
  if [[ -n "${COMPONENTS:-}" ]]; then echo "$COMPONENTS"; return; fi
  case "$1" in
    osim) echo "0 1 2 3" ;;   # all four leaf scorers (c1 is a known dead axis)
    peri) echo "0 1" ;;       # both leaf scorers
    *) echo "" ;;
  esac
}

# same translation as stage 1 (01_train_guides.sh); duplicated rather than
# sourced so this stage stays runnable standalone.
guide_flags () {
  case "$1" in
    hidden) echo "" ;;
    base)   echo "--no_use_hidden_guide" ;;
    *) echo "UNKNOWN_GUIDE_$1" ;;
  esac
}
replay_flags () {
  case "$1" in
    on)  echo "--use_replay --replay_fraction 0.25 --replay_strategy reward" ;;
    off) echo "" ;;
    *) echo "UNKNOWN_REPLAY_$1" ;;
  esac
}

RAN=0; SKIPPED=0
START_TS=$(date +%s)

launch () {
  local name="$1" cmd="$2"
  hr; echo "[run] $name"; echo "$cmd"; hr
  [[ "$DRY" == "1" ]] && return 0
  throttle
  local gpu=$(( RAN % NUM_GPUS ))
  (
    CUDA_VISIBLE_DEVICES="$gpu" eval "$cmd" > "${LOGDIR}/${name}.log" 2>&1
    RC=$?
    case $RC in
      0)  ;;
      17) echo "[stall] $name hit the hang guard (see ${LOGDIR}/${name}.log)" ;;
      18) echo "[timelimit] $name stopped at ${MAX_TRAIN_HOURS}h (see ${LOGDIR}/${name}.log)" ;;
      *)  echo "[warn] $name exited $RC (see ${LOGDIR}/${name}.log)" ;;
    esac
  ) &
  echo "[launch] $name (pid $!, gpu $gpu, active=$(( $(jobs -r -p | wc -l) )))"
  RAN=$((RAN+1)); sleep 2
}

# ------------------------- per-component guides ------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "components" ]]; then
  # REWARDS set -> loop the multi-bench grid; empty -> the single BENCH/BENCH_TAG
  if [[ -n "$REWARDS" ]]; then
    REWARD_LIST="$REWARDS"
  else
    REWARD_LIST="$BENCH_TAG"
  fi
  say "component guides: rewards='$REWARD_LIST' guides='$GUIDES' objectives='$OBJECTIVES'"
  say "                   replay='$REPLAYS' betas='$BETAS' seeds='${SEEDS:-<none>}'"
  for tag in $REWARD_LIST; do
    if [[ -n "$REWARDS" ]]; then
      full=$(bench_full "$tag")
      comps=$(components_for "$tag")
      if [[ -z "$full" || -z "$comps" ]]; then
        echo "[fatal] no bench/components mapping for reward '$tag'; add one to" \
             "bench_full()/components_for() or set COMPONENTS explicitly" >&2
        exit 1
      fi
    else
      full="$BENCH"; comps="${COMPONENTS:-0 2 3}"
    fi
    for comp in $comps; do
      for guide in $GUIDES; do
        for obj in $OBJECTIVES; do
          for replay in $REPLAYS; do
            for beta in $BETAS; do
              for seed in ${SEEDS:-""}; do
                NAME="compose-${tag}-c${comp}-${guide}-${obj}-b${beta}"
                [[ "$replay" == "on" ]] && NAME="${NAME}-replay_on"
                SEED_ARG=""
                if [[ -n "$seed" ]]; then
                  NAME="${NAME}-s${seed}"
                  SEED_ARG="--seed ${seed}"
                fi
                if compgen -G "${CKPT_ROOT}/${NAME}/checkpoints/*.ckpt" > /dev/null; then
                  echo "[skip] $NAME (checkpoint exists)"; SKIPPED=$((SKIPPED+1)); continue
                fi
                launch "$NAME" "$PY gflow.py \
                  --name ${NAME} \
                  --quetzal_ckpt ${QUETZAL_CKPT} \
                  --objective ${obj} \
                  --reward guacamol_component \
                  --reward_benchmark ${full} \
                  --reward_component ${comp} \
                  --reward_beta ${beta} \
                  ${SEED_ARG} \
                  ${GUARD_FLAGS} \
                  $(guide_flags "$guide") \
                  $(replay_flags "$replay") \
                  --max_epochs ${MAX_EPOCHS} \
                  --steps_per_epoch ${STEPS} \
                  --eval_n 0 --final_n 0 \
                  --hist_every_n_epochs 0 \
                  --no_fcd_enabled --no_eval_base"
              done
            done
          done
        done
      done
    done
  done
fi

# --------------------------- atom-stability ----------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "stability" ]]; then
  say "atom-stability guides (hidden vs base control), beta 1 and 10"
  for guide in hidden base; do
    for beta in 1 10; do
      NAME="stability-geom-${guide}-db-b${beta}"
      if compgen -G "${CKPT_ROOT}/${NAME}/checkpoints/*.ckpt" > /dev/null; then
        echo "[skip] $NAME (checkpoint exists)"; SKIPPED=$((SKIPPED+1)); continue
      fi
      GFLAGS=""
      [[ "$guide" == "base" ]] && GFLAGS="--no_use_hidden_guide"
      launch "$NAME" "$PY gflow.py \
        --name ${NAME} \
        --quetzal_ckpt ${QUETZAL_CKPT} \
        --objective db \
        --reward atom_stability \
        --reward_beta ${beta} \
        ${GFLAGS} \
        --dataset geom \
        --max_epochs ${MAX_EPOCHS} \
        --steps_per_epoch ${STEPS} \
        --eval_n 500 --final_n 0 \
        --hist_every_n_epochs 0 \
        --no_fcd_enabled"
    done
  done
fi

wait
END_TS=$(date +%s)
hr
say "stage 2 done: ran=$RAN skipped=$SKIPPED elapsed $(( (END_TS-START_TS)/60 )) min"
say "components feed scripts/05_dump_composed.sh; stability feeds scripts/04_dump_guides.sh"
hr
