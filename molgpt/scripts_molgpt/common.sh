#!/bin/bash
# Sourced by every scripts_molgpt/*.sh driver. Mirrors quetzal_gfn's
# scripts/common.sh (see scripts/README.md there) with MOLGPT_CKPT in place
# of QUETZAL_CKPT and no hang-guard flags wired in yet -- see
# gflow_molgpt.py's module docstring, "No hang-guard", for why.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/molgpt_gfn${PYTHONPATH:+:${PYTHONPATH}}"
cd "$REPO_ROOT"

_QUETZAL_ENV_PY="/root/miniconda3/envs/quetzal/bin/python"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$_QUETZAL_ENV_PY" ]]; then
  # Plain `python` on PATH resolves to base conda here, not the `quetzal` env
  # torch/rdkit/guacamol are actually installed in -- prefer the env directly
  # rather than relying on `conda activate` having run in whatever shell
  # invoked this (nohup in particular starts a fresh, non-interactive shell
  # with no conda hook sourced).
  PY="$_QUETZAL_ENV_PY"
else
  PY="python"
fi
# GEOM-Drugs retrain (2026-09-04, author instruction: both new legs must
# train on GEOM-Drugs, matching Quetzal's own corpus, with Quetzal's own
# atom vocabulary -- see ../../shared_data/PROVENANCE.md). The old
# GuacaMol-pretrained checkpoint stays on disk but is no longer the default.
MOLGPT_CKPT="${MOLGPT_CKPT:-checkpoints/molgpt_geom_drugs_unconditional.pt}"
CKPT_ROOT="${CKPT_ROOT:-logs/molgpt-gfn}"
RESULTS_ROOT="${RESULTS_ROOT:-results/molgpt}"
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

# Lock-based GPU allocation, NOT round-robin-by-launch-order -- switched
# 2026-09-04 after the identical round-robin scheme caused 11 of 24 g2pt
# sweep runs to OOM (two temporally-overlapping jobs assigned the same
# physical GPU, since round-robin-by-launch-order only avoids collisions if
# every job takes the same wall-clock time -- false whenever db/rtb or
# different rewards finish at different speeds). molgpt's own sweep never
# hit this in practice (its jobs are light enough that two sharing a GPU
# didn't exceed 32GB), but it's the same latent bug -- fixed here too rather
# than leaving a working-by-luck script in place for future runs.
# `mkdir` is atomic, so this is a correct mutex regardless of job timing.
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
  if [[ ! -f "$MOLGPT_CKPT" ]]; then
    echo "[FATAL] MOLGPT_CKPT not found: $MOLGPT_CKPT" >&2
    echo "        Download the pretrained weights (see ../README.md, 'Blocked')" >&2
    echo "        or point MOLGPT_CKPT at wherever you put them." >&2
    exit 1
  fi
}

resolve_ckpt() {
  # $1 = run name. Prefers checkpoints/last.ckpt, else the newest *.ckpt.
  local name="$1"
  local dir="${CKPT_ROOT}/${name}/checkpoints"
  if [[ -f "${dir}/last.ckpt" ]]; then
    echo "${dir}/last.ckpt"
  else
    ls -t "${dir}"/*.ckpt 2>/dev/null | head -1
  fi
}
