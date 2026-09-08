#!/bin/bash
# Full FUDGE pipeline for Quetzal: data -> train -> eval, all 4 rewards,
# reduced rollout budget (n=500 vs the pilots' 5000, see decisions.md),
# sequential on whichever single GPU CUDA_VISIBLE_DEVICES pins this to.
set -uo pipefail
cd /workspace/quetzal/quetzal_gfn
export PYTHONPATH="/workspace/quetzal/quetzal_gfn${PYTHONPATH:+:${PYTHONPATH}}"
PY=/root/miniconda3/envs/quetzal/bin/python
mkdir -p fudge_data logs/drivers results/fudge_dumps results/fudge_flips

REWARDS=(osim peri fexo nitrogen)
for r in "${REWARDS[@]}"; do
  echo "=== [$r] data ==="
  $PY fudge_data.py --reward "$r" --n 500 --chunk 64 \
    --out "fudge_data/${r}.pt" || { echo "[FATAL] fudge_data $r failed"; exit 1; }

  echo "=== [$r] train ==="
  $PY train_fudge.py --data "fudge_data/${r}.pt" --name "fudge-${r}" \
    || { echo "[FATAL] train_fudge $r failed"; exit 1; }

  CKPT="logs/quetzal-gfn/fudge-${r}/checkpoints/last.ckpt"
  echo "=== [$r] final_dump ==="
  $PY final_dump.py --ckpt "$CKPT" --n 500 --chunk 500 --guide_source ema --no_fcd \
    --out_dir "results/fudge_dumps/${r}" || { echo "[FATAL] final_dump $r failed"; exit 1; }

  echo "=== [$r] flip ==="
  $PY ablations/single_flip_ablation.py --ckpt "$CKPT" --label "fudge-${r}" \
    --report_tag "fudge-${r}" --n_traj 200 --chunk 200 --guide_source ema \
    --out_dir results/fudge_flips || { echo "[FATAL] flip $r failed"; exit 1; }
done
echo "=== QUETZAL FUDGE PIPELINE COMPLETE ==="
