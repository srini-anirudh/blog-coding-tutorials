#!/bin/bash
# Run every GPU script in order. 03 and 04 measure the ceilings that 05-09 compare against.
# Usage (from the repository root, with one CUDA GPU visible): bash labs/why-fast-gpus-still-wait-for-memory/scripts/run_gpu.sh
set -e
PY=${PY:-python}
D=$(dirname "$0")
for s in 03_bandwidth 04_compute_ceilings 05_roofline_sweep 06_gemm_shapes 07_elementwise_fusion 08_attention 09_optimizers; do
  echo "=== $s"
  $PY "$D/$s.py"
done
