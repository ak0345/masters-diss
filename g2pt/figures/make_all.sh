#!/bin/bash
# Regenerate every figure in this directory. Mirrors ../../figures/make_all.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for f in make_fig*.py; do
  echo "=== $f ==="
  python3 "$f"
done
