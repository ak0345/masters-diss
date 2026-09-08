#!/usr/bin/env bash
# =============================================================================
# run_study_compose.sh -- the composition track (stages 2 + 5), supervised, in
# one invocation. Companion to run_study.sh/run_study2.sh, which train and dump
# the single-guide sweep; this one trains the per-component teachers and
# composes them, for the dissertation's composition chapter.
#
# THE GRID
#   components (stage 2)   2 rewards {osim, peri}
#                          x 2 architectures {hidden, base}
#                          x 2 objectives {db, rtb}
#                          x replay off only
#                          x beta 10 only
#                          x 2 training seeds {0, 42}
#                          x components: osim has 4 (0..3), peri has 2 (0..1)
#                          = (4+2) x 2 x 2 x 2 = 48 guide-training runs
#                          500 optimiser steps at batch 128, matching the
#                          single-guide sweep's per-run budget.
#
#   composed dumps (stage 5)   one composed sampler per (reward, guide,
#                          objective, training seed) = 2x2x2x2 = 16 configs,
#                          x 3 operators {linear, product, harmonic}
#                          x ONE dump (sampling) seed -- the training-seed axis
#                          already gives run-to-run variance, so a second,
#                          independent resampling seed adds little.
#
# ORDER
#   2 components   COMPONENT_PARALLEL concurrent runs
#   5 compose      COMPOSE_PARALLEL concurrent runs
#
# SUPERVISION -- same as run_study.sh: GUARD_STALL_MINUTES stalls a run at 17,
# MAX_TRAIN_HOURS stops it at 18; both leave it resumable. Each stage is
# re-invoked up to MAX_RETRIES times; every stage is independently resumable
# (a run whose checkpoint or dump_summary.json exists is skipped), so a retry
# only repeats what did not finish.
#
# USAGE
#   DRY=1 bash scripts/run_study_compose.sh       # print every command, run none
#   bash scripts/run_study_compose.sh              # the whole composition track
#   STAGES=2 bash scripts/run_study_compose.sh      # components only
#   STAGES=5 bash scripts/run_study_compose.sh      # compose/dump only (needs 2 done)
#
#   MAX_TRAIN_HOURS=3 nohup bash scripts/run_study_compose.sh > compose_study.log 2>&1 &
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------ the study grid -------------------------------
export TRAIN_SEEDS="${TRAIN_SEEDS:-0 42}"   # component guides' training seed
export REWARDS="${REWARDS:-osim peri}"
export GUIDES="${GUIDES:-hidden base}"
export OBJECTIVES="${OBJECTIVES:-db rtb}"
export REPLAYS="${REPLAYS:-off}"
# BETA (singular) is what stage 5 reads -- a composed sampler mixes K
# components trained at ONE shared beta, so the grid is single-valued here.
# BETAS (plural) is stage 2's axis name; keep them in sync.
export BETA="${BETA:-10}"
export BETAS="${BETAS:-$BETA}"
export MAX_EPOCHS="${MAX_EPOCHS:-5}"        # x STEPS = 500 optimiser steps
export STEPS="${STEPS:-100}"
export BSZ="${BSZ:-128}"

export GUARD_STALL_MINUTES="${GUARD_STALL_MINUTES:-15}"
export GUARD_REWARD_TIMEOUT="${GUARD_REWARD_TIMEOUT:-20}"
export MAX_TRAIN_HOURS="${MAX_TRAIN_HOURS:-3}"
export NUM_GPUS="${NUM_GPUS:-3}"
export DRY="${DRY:-0}"

# Per-stage concurrency.
COMPONENT_PARALLEL="${COMPONENT_PARALLEL:-5}"
COMPOSE_PARALLEL="${COMPOSE_PARALLEL:-3}"      # dumps are heavier; keep <= NUM_GPUS

# Dump-side: one sampling seed, 5k molecules -- the training-seed axis already
# gives the run-to-run variance that matters for the dissertation.
export DUMP_N="${DUMP_N:-5000}"
export DUMP_SEED="${DUMP_SEED:-0}"
export CONV_PROCS="${CONV_PROCS:-16}"
export CONV_TIMEOUT="${CONV_TIMEOUT:-90}"

MAX_RETRIES="${MAX_RETRIES:-3}"
STAGES="${STAGES:-2 5}"
BACKOFF="${BACKOFF:-30}"

source "$HERE/common.sh"

# --------------------------------- driver ------------------------------------
run_stage () {
  local label="$1"; shift
  local attempt=0
  while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt+1))
    hr
    say "STAGE ${label}  (attempt ${attempt}/${MAX_RETRIES})"
    hr
    if "$@"; then
      say "STAGE ${label} finished"
      return 0
    fi
    say "STAGE ${label} exited non-zero on attempt ${attempt}"
    (( attempt < MAX_RETRIES )) && sleep "$BACKOFF"
  done
  say "STAGE ${label} still failing after ${MAX_RETRIES} attempts; continuing"
  return 1
}

START=$SECONDS
say "compose study grid: rewards='${REWARDS}' guides='${GUIDES}' objectives='${OBJECTIVES}'"
say "                     betas='${BETAS}' replay='${REPLAYS}' train_seeds='${TRAIN_SEEDS}'"
say "                     ${MAX_EPOCHS}x${STEPS} steps at batch ${BSZ}, guard ${GUARD_STALL_MINUTES} min,"
say "                     max_train_hours=${MAX_TRAIN_HOURS}, up to ${MAX_RETRIES} attempts per stage"
say "                     stage order '${STAGES}' | components x${COMPONENT_PARALLEL}, compose x${COMPOSE_PARALLEL}"
say "                     dump: n=${DUMP_N} seed=${DUMP_SEED} (one seed per configuration)"

for s in $STAGES; do
  case "$s" in
    2) run_stage "2 components" \
         env MAX_PARALLEL="$COMPONENT_PARALLEL" SEEDS="$TRAIN_SEEDS" \
             bash "$HERE/02_train_components.sh" components ;;
    5) run_stage "5 compose" \
         env MAX_PARALLEL="$COMPOSE_PARALLEL" N="$DUMP_N" SEEDS="$DUMP_SEED" \
             bash "$HERE/05_dump_composed.sh" ;;
    *) say "unknown stage '$s' (valid: 2 5)" ;;
  esac
done

hr
say "compose study finished in $(( (SECONDS-START)/3600 ))h $(( ((SECONDS-START)%3600)/60 ))m"
say "master table: ${RESULTS_ROOT}/dumps/_aggregate/master_table.csv"
hr
