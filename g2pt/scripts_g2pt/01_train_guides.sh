#!/bin/bash
# Train the pilot grid for the G2PT (2D) leg. Mirrors scripts_molgpt/01_train_guides.sh's
# env-var interface and conventions (DRY=1 prints commands and runs nothing; a
# run is skipped if its checkpoint directory already has a *.ckpt in it;
# run-directory naming is sweep-<reward>-<guide>-<objective>-replay_off-b<beta>-s<seed>).
#
# Same scope discipline as the other two legs -- see plan_molgpt.md and
# plan_graphinvent.md (this leg replaces GraphINVENT but keeps its grid shape,
# see papers/current/decisions.md, 2026-09-03):
#   guides:     hidden, base
#   objectives: db, rtb
#   replay:     off only
#   beta:       10 only
#   seeds:      3
#   tasks:      nitrogen first, then one GuacaMol MPO task
#
# MAX_LEN=300 (not the GEOM-Drugs checkpoint's block_size=700): the training
# corpus's own encoded-sequence-length distribution (233,155 molecules,
# shared_data/, same BFS grammar) has p99=277, p999=316 -- 300 sits between
# those, ~99.5% coverage, while staying under half of block_size=700 (the
# same safety margin the old GuacaMol-vocab checkpoint's 300/614 ratio used,
# and that checkpoint's own OOM ceiling was at the full block_size, not at
# 300 -- see g2pt_gfn/config.py's `max_len` comment and vendor/PROVENANCE.md).
# Raising this back toward 700 without re-measuring risks repeating that OOM.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
require_prior

GUIDES="${GUIDES:-hidden base}"
OBJECTIVES="${OBJECTIVES:-db rtb}"
BETAS="${BETAS:-10}"
SEEDS="${SEEDS:-0 42 100}"
REWARDS="${REWARDS:-nitrogen osim peri fexo}"   # peri/fexo folded in 2026-09-04, same
                                                  # reasoning as scripts_molgpt/01_train_guides.sh

MAX_EPOCHS="${MAX_EPOCHS:-5}"
STEPS="${STEPS:-100}"
BSZ="${BSZ:-128}"
MAX_LEN="${MAX_LEN:-300}"
LOGDIR="${LOGDIR:-${LOG_ROOT}/guides_g2pt}"
mkdir -p "$LOGDIR"

reward_flags() {
  # peri/fexo added post-hoc, author instruction 2026-09-04 -- same
  # benchmark names as the molgpt leg's reward_flags() and the parent
  # repo's scripts/01_train_guides.sh.
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
    hidden) echo "" ;;
    base)   echo "--no-use_hidden_guide" ;;
    *) echo "[FATAL] unknown guide '$1'" >&2; exit 1 ;;
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

          CMD="$PY g2pt_gfn/gflow_g2pt.py --name $NAME --model_name_or_path $G2PT_MODEL \
            --ckpt_root $CKPT_ROOT \
            --objective $obj --reward_beta $beta --seed $seed \
            --max_epochs $MAX_EPOCHS --steps_per_epoch $STEPS --bsz $BSZ --max_len $MAX_LEN \
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
