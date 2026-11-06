#!/usr/bin/env bash
# Runs every GPU measurement of this lab on one GPU, in order. About 10 minutes on one H100.
# Run from the repository root with exactly one GPU visible. A failing script does not stop the others.
set -u
LAB=labs/the-mechanics-of-llm-inference
PY=${PY:-python}
for s in "03_roofline_gemm roofline_gemm_h100" "04_decode_wall decode_wall_h100" "05_batching batching_h100" \
         "06_quantization quantization_h100" "07_speculative speculative_h100"; do
  set -- $s
  echo "=== $1"
  start=$SECONDS
  $PY $LAB/scripts/$1.py --out $LAB/results/$2.txt || echo "FAILED: $1"
  echo "--- $1 took $((SECONDS - start)) s"
done
echo "=== pytest -m gpu"
{ $PY -c "import sys; sys.path.insert(0, '$LAB'); from inferlab.timing import hardware_line; print(hardware_line())"
  echo "# Command: python -m pytest $LAB -m gpu"; echo
  $PY -m pytest $LAB -m gpu -o addopts="--import-mode=importlib" -q -p no:cacheprovider 2>&1; } | tee $LAB/results/pytest_gpu.txt
