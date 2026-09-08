#!/bin/bash
# Full FUDGE pipeline for G2PT: data -> train -> eval, all 4 rewards,
# sequential on whichever single GPU CUDA_VISIBLE_DEVICES pins this to.
set -uo pipefail
cd /workspace/quetzal/quetzal_gfn/g2pt
PY=/root/miniconda3/envs/quetzal/bin/python
mkdir -p results/g2pt/fudge_data logs/drivers results/g2pt/fudge_dumps results/g2pt/fudge_flips

REWARDS=(osim peri fexo nitrogen)
for r in "${REWARDS[@]}"; do
  echo "=== [$r] data ==="
  $PY g2pt_gfn/fudge_data_g2pt.py --reward "$r" --n 5000 --chunk 128 \
    --out "results/g2pt/fudge_data/${r}.pt" || { echo "[FATAL] fudge_data $r failed"; exit 1; }

  echo "=== [$r] train ==="
  $PY g2pt_gfn/train_fudge_g2pt.py --data "results/g2pt/fudge_data/${r}.pt" \
    --name "fudge-${r}" || { echo "[FATAL] train_fudge $r failed"; exit 1; }

  CKPT="logs/g2pt-gfn/fudge-${r}/checkpoints/last.ckpt"
  echo "=== [$r] final_dump ==="
  $PY g2pt_gfn/final_dump_g2pt.py --ckpt "$CKPT" --n 5000 --guide_source ema \
    --sample_temp 1.0 --out_dir "results/g2pt/fudge_dumps/${r}" || { echo "[FATAL] final_dump $r failed"; exit 1; }

  echo "=== [$r] flip ==="
  $PY g2pt_gfn/flip_g2pt.py --ckpt "$CKPT" --label "fudge-${r}" \
    --out_dir results/g2pt/fudge_flips --n_traj 500 || { echo "[FATAL] flip $r failed"; exit 1; }
done
echo "=== G2PT FUDGE PIPELINE COMPLETE ==="
