#!/bin/bash
# Full FUDGE pipeline for MolGPT: data -> train -> eval, all 4 rewards,
# sequential on whichever single GPU CUDA_VISIBLE_DEVICES pins this to.
set -uo pipefail
cd /workspace/quetzal/quetzal_gfn/molgpt
PY=/root/miniconda3/envs/quetzal/bin/python
mkdir -p results/molgpt/fudge_data logs/drivers results/molgpt/fudge_dumps results/molgpt/fudge_flips

REWARDS=(osim peri fexo nitrogen)
for r in "${REWARDS[@]}"; do
  echo "=== [$r] data ==="
  $PY molgpt_gfn/fudge_data_molgpt.py --reward "$r" --n 5000 --chunk 256 \
    --out "results/molgpt/fudge_data/${r}.pt" || { echo "[FATAL] fudge_data $r failed"; exit 1; }

  echo "=== [$r] train ==="
  $PY molgpt_gfn/train_fudge_molgpt.py --data "results/molgpt/fudge_data/${r}.pt" \
    --name "fudge-${r}" || { echo "[FATAL] train_fudge $r failed"; exit 1; }

  CKPT="logs/molgpt-gfn/fudge-${r}/checkpoints/last.ckpt"
  echo "=== [$r] final_dump ==="
  $PY molgpt_gfn/final_dump_molgpt.py --ckpt "$CKPT" --n 5000 --guide_source ema \
    --out_dir "results/molgpt/fudge_dumps/${r}" || { echo "[FATAL] final_dump $r failed"; exit 1; }

  echo "=== [$r] flip ==="
  $PY molgpt_gfn/flip_molgpt.py --ckpt "$CKPT" --label "fudge-${r}" \
    --out_dir results/molgpt/fudge_flips --n_traj 500 || { echo "[FATAL] flip $r failed"; exit 1; }
done
echo "=== MOLGPT FUDGE PIPELINE COMPLETE ==="
