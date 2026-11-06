#!/bin/bash
# Every GPU script, one after another, on one GPU, then the gpu tests. Outputs go to results/.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
for s in 01_busy_fraction 02_utilisation_lies 03_roofline 06_memory 07_decode_batching 08_whack_a_mole; do
  echo "=== $s"
  $PY scripts/$s.py
done
echo "=== gpu tests"
$PY -m pytest tests/test_gpu.py -m gpu -o addopts="--import-mode=importlib" -q -p no:cacheprovider 2>&1 | tail -5
