#!/bin/bash
# Train the pilot grid. Mirrors quetzal_gfn's scripts/01_train_guides.sh env-var
# interface and conventions (DRY=1 prints commands and runs nothing; a run is
# skipped if its checkpoint directory already has a *.ckpt in it; run-directory
# naming is sweep-<reward>-<guide>-<objective>-replay_off-b<beta>-s<seed>).
#
# plan_molgpt.md fixes this pilot's scope deliberately narrow -- READ THAT FILE
# before widening any of these axes:
#   guides:     hidden, base            (no tempgain)
#   objectives: db, rtb
#   replay:     off only
#   beta:       10 only
#   seeds:      3 training seeds
#   tasks:      nitrogen first (fast sanity check), then one GuacaMol MPO task
#
# That's 2 x 2 x 3 x 2 = 24 runs. Overridable via env vars below for a debug
# run (e.g. one config, one epoch) -- NOT for silently expanding the grid.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_prior

GUIDES="${GUIDES:-hidden base}"
OBJECTIVES="${OBJECTIVES:-db rtb}"
BETAS="${BETAS:-10}"
SEEDS="${SEEDS:-0 42 100}"
REWARDS="${REWARDS:-nitrogen osim peri fexo}"   # nitrogen first (plan_molgpt.md's fast
                                                  # sanity check); peri/fexo folded in
                                                  # 2026-09-04 (author instruction) rather
                                                  # than run as a separate follow-up pass,
                                                  # now that EMA requires a full redo anyway

MAX_EPOCHS="${MAX_EPOCHS:-5}"
STEPS="${STEPS:-100}"
BSZ="${BSZ:-128}"
LOGDIR="${LOGDIR:-${LOG_ROOT}/guides}"
mkdir -p "$LOGDIR"

reward_flags() {
  # peri/fexo added post-hoc, author instruction 2026-09-04 (beyond
  # plan_molgpt.md's original osim-only scope) -- same benchmark names as
  # the parent repo's scripts/01_train_guides.sh reward_flags().
  case "$1" in
    osim)     echo "--reward guacamol --reward_smiles hard_osimertinib" ;;
    peri)     echo "--reward guacamol --reward_smiles perindopril_rings" ;;
    fexo)     echo "--reward guacamol --reward_smiles hard_fexofenadine" ;;
    nitrogen) echo "--reward nitrogen_count" ;;
    *) echo "[FATAL] unknown reward '$1'" >&2; exit 1 ;;
  esac
}

guide_flags() {
  case "$1" in
    hidden) echo "" ;;                        # use_hidden_guide defaults True
    base)   echo "--no-use_hidden_guide" ;;
    *) echo "[FATAL] unknown guide '$1' -- this pilot only wires up hidden/base (plan_molgpt.md, no tempgain)" >&2; exit 1 ;;
  esac
}

COUNT=0
SKIPPED=0
for reward in $REWARDS; do
  RFLAGS="$(reward_flags "$reward")" || exit 1
  for guide in $GUIDES; do
    GFLAGS="$(guide_flags "$guide")" || exit 1
    for obj in $OBJECTIVES; do
      for beta in $BETAS; do
        for seed in $SEEDS; do
          COUNT=$((COUNT+1))
          NAME="sweep-${reward}-${guide}-${obj}-replay_off-b${beta}-s${seed}"

          if compgen -G "${CKPT_ROOT}/${NAME}/checkpoints/*.ckpt" > /dev/null; then
            echo "[skip $COUNT] $NAME (checkpoint exists)"
            SKIPPED=$((SKIPPED+1))
            continue
          fi

          CMD="$PY molgpt_gfn/gflow_molgpt.py --name $NAME --molgpt_ckpt $MOLGPT_CKPT \
            --ckpt_root $CKPT_ROOT \
            --objective $obj --reward_beta $beta --seed $seed \
            --max_epochs $MAX_EPOCHS --steps_per_epoch $STEPS --bsz $BSZ \
            $RFLAGS $GFLAGS"

          hr
          echo "[run $COUNT] $NAME"
          echo "$CMD"
          hr
          [[ "$DRY" == "1" ]] && continue

          throttle
          ( GPU="$(acquire_gpu)"
            trap 'release_gpu "$GPU"' EXIT
            say "[start] $NAME (GPU $GPU)"
            export CUDA_VISIBLE_DEVICES="$GPU"
            eval "$CMD" > "${LOGDIR}/${NAME}.log" 2>&1
            code=$?
            if [[ $code -eq 0 ]]; then
              say "[done] $NAME (GPU $GPU)"
            else
              say "[warn] $NAME exited $code (see ${LOGDIR}/${NAME}.log)"
            fi
          ) &
          sleep 2
        done
      done
    done
  done
done

wait
echo "ran/queued $((COUNT-SKIPPED)) of $COUNT configs, skipped $SKIPPED already-done"
