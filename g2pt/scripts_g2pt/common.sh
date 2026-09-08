#!/bin/bash
# Sourced by every scripts_g2pt/*.sh driver. Mirrors scripts_molgpt/common.sh
# (see that file's comments) with G2PT_MODEL in place of MOLGPT_CKPT -- G2PT
# has no local checkpoint file to point at, it's a HuggingFace model id
# resolved by `transformers.AutoModelForCausalLM.from_pretrained` on first use
# and cached under ~/.cache/huggingface, so there is no local-file
# `require_prior`-style existence check here (see `require_prior` below for
# what it checks instead).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/g2pt_gfn${PYTHONPATH:+:${PYTHONPATH}}"
cd "$REPO_ROOT"

_QUETZAL_ENV_PY="/root/miniconda3/envs/quetzal/bin/python"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$_QUETZAL_ENV_PY" ]]; then
  PY="$_QUETZAL_ENV_PY"
else
  PY="python"
fi

# GEOM-Drugs retrain (2026-09-04, author instruction: both new legs must
# train on GEOM-Drugs with Quetzal's own atom vocabulary, since Quetzal
# itself can't be rerun -- see ../../shared_data/PROVENANCE.md). No HF
# checkpoint exists for this dataset/vocab -- G2PT_MODEL now points at a
# local directory (checkpoints/g2pt_geom_drugs_hf/, mirroring molgpt's
# checkpoints/ convention), copied there from vendor/G2PT/results/
# geom_drugs-small-bfs/hf/ (convert_to_hf.py's output on the from-scratch
# nanoGPT checkpoint) by run_pipeline.sh's Stage 1 -- not a HuggingFace Hub
# id. AutoModelForCausalLM.from_pretrained accepts a local directory
# identically to a Hub id, no code change needed downstream.
G2PT_MODEL="${G2PT_MODEL:-${REPO_ROOT}/checkpoints/g2pt_geom_drugs_hf}"
CKPT_ROOT="${CKPT_ROOT:-logs/g2pt-gfn}"
RESULTS_ROOT="${RESULTS_ROOT:-results/g2pt}"
LOG_ROOT="${LOG_ROOT:-logs/drivers}"

DRY="${DRY:-0}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NUM_GPUS="${NUM_GPUS:-1}"
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
MAX_PARALLEL="${MAX_PARALLEL:-$NUM_GPUS}"

say() { echo "[$(date +%H:%M:%S)] $*"; }
hr() { printf '%s\n' "------------------------------------------------------------"; }

throttle() {
  while [[ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    sleep 2
  done
}

# Lock-based GPU allocation, NOT round-robin-by-launch-order. A naive
# `(count-1) % NUM_GPUS` scheme (what this used to be) only avoids two jobs
# sharing a GPU if every job takes the same wall-clock time -- false here:
# db vs rtb and different rewards finish at very different speeds, so two
# temporally-overlapping jobs legitimately got assigned the same physical
# GPU by round-robin and OOM'd fighting over one card while another sat idle
# -- confirmed 2026-09-04, 11 of 24 g2pt sweep runs failed exactly this way
# (see g2pt/vendor/PROVENANCE.md). `mkdir` is atomic, so this is a correct
# mutex regardless of job timing.
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/tmp/quetzal_gfn_gpu_locks}"
# Fixed 2026-09-04: this used to `rm -rf` the lock dir on every source, to
# clear stale locks from a prior crashed invocation. That's unsafe now that
# molgpt's and g2pt's pipelines run *concurrently* and share this same
# directory (GEOM-Drugs retrain for both legs, launched together) -- a
# driver sourcing this file while the other leg's pipeline genuinely holds a
# lock would silently delete that lock out from under it. Just ensure the
# dir exists; a real stale lock from a crashed run needs a human to `rm` it
# (or reboot the box), not an automatic wipe that can't tell "stale" from
# "another pipeline is using this GPU right now".
mkdir -p "$GPU_LOCK_DIR"

# GPU_OFFSET (2026-09-04): confirmed 2026-09-04 that two *separate* driver
# scripts racing for the same shared lock pool starves one of them --
# molgpt's sweep won every single lock-release race against g2pt's for ~20
# minutes straight (g2pt made zero progress) despite both using the same
# 2s-poll acquire_gpu. Root cause not fully pinned down (plausibly molgpt's
# already-blocked `throttle` loop reacting to a same-script release faster
# than g2pt's independently-phased waiters ever could), but the robust fix
# isn't smarter racing, it's not racing at all: GPU_OFFSET plus NUM_GPUS lets
# each leg be pinned to a disjoint slice of the physical GPUs (e.g. molgpt
# NUM_GPUS=1 GPU_OFFSET=0, g2pt NUM_GPUS=2 GPU_OFFSET=1), so `acquire_gpu`
# only ever contends with itself. Defaults to 0 (whole pool, old behaviour)
# when only one pipeline is running.
GPU_OFFSET="${GPU_OFFSET:-0}"

acquire_gpu() {
  # Blocks until a GPU is free, claims it, prints its index. Caller MUST
  # call `release_gpu <idx>` when done -- e.g. `trap 'release_gpu "$GPU"' EXIT`
  # in the subshell that used it, so the lock releases even on a crash/OOM.
  while true; do
    for ((i=GPU_OFFSET; i<GPU_OFFSET+NUM_GPUS; i++)); do
      if mkdir "${GPU_LOCK_DIR}/gpu_${i}.lock" 2>/dev/null; then
        echo "$i"
        return 0
      fi
    done
    sleep 2
  done
}

release_gpu() {
  rmdir "${GPU_LOCK_DIR}/gpu_${1}.lock" 2>/dev/null
}

require_prior() {
  # Confirms the checkpoint resolves and generates, without a guide, once --
  # equivalent to molgpt's file-existence check, but since there's no local
  # file, this actually runs the step-1 sanity check (cheap: one HF download
  # if not cached, one small unconditional batch). Skips if a marker from a
  # previous successful check exists, so re-invoking the sweep doesn't
  # re-verify every time.
  local marker="${CKPT_ROOT}/.prior_verified_${G2PT_MODEL//\//_}"
  if [[ -f "$marker" ]]; then
    return 0
  fi
  say "verifying G2PT prior ($G2PT_MODEL) loads and generates..."
  # batch_size 64, not 16 (fixed 2026-09-05): at the model's real ~24.5%
  # validity (confirmed once _decode's <boc>-stripping bug was fixed -- see
  # decisions.md), a 16-sample draw is far too small to reliably clear even
  # a 20% threshold (binomial(16, 0.245) has a ~1.7-sample std, so a single
  # unlucky draw of 2/16=12.5% is ordinary noise, not a real regression) --
  # this exact flakiness made a live pipeline run fail Stage 3's prior check
  # and (a separate bug, also fixed) cascade silently through every
  # remaining stage with zero trained configs. batch_size 64 matches the
  # sanity check every run_pipeline.sh invocation already does inline in
  # Stage 2, which has never spuriously failed.
  if ! "$PY" g2pt_gfn/g2pt_prior.py --model_name_or_path "$G2PT_MODEL" \
       --batch_size 64 --min_valid_frac 0.15 > /tmp/g2pt_prior_check.log 2>&1; then
    echo "[FATAL] G2PT prior failed its sanity check -- see /tmp/g2pt_prior_check.log" >&2
    exit 1
  fi
  mkdir -p "$CKPT_ROOT"
  touch "$marker"
  say "prior verified"
}
