#!/bin/bash
# Build results/molgpt/_aggregate/master_table.csv from every final-dump
# summary and flip report on disk. Cheap, re-runs every time (same
# discipline as the parent repo's harvest stages -- see scripts/README.md's
# resumability table: "re-runs every time" is deliberate for cheap
# CPU-only aggregation that should pick up anything finished since the last pass).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

"$PY" molgpt_gfn/aggregate_molgpt.py \
  --dumps_root "${RESULTS_ROOT}/dumps" \
  --flips_root "${RESULTS_ROOT}/flips" \
  --out_dir "${RESULTS_ROOT}/_aggregate"
