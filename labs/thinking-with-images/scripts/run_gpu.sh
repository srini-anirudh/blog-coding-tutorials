#!/usr/bin/env bash
# Every model experiment, in order, on one GPU. About 30 min on one H100 80GB.
# Run from the repository root: bash labs/thinking-with-images/scripts/run_gpu.sh
# Each script writes results/<name>.txt and .json. A failing script does not stop the others.
set -u
PY=${PY:-python}
S=labs/thinking-with-images/scripts
export TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error
for cmd in "02_lost_detail.py --n 100" "03_probes.py --n 100" "04_views_vs_samples.py --n 50 --k 16" \
           "05_drift.py --n 12 --gen-tokens 1024" "06_search.py --n 60"; do
  start=$SECONDS
  echo "=== $cmd"
  $PY $S/$cmd || echo "FAILED: $cmd"
  echo "--- $cmd took $((SECONDS - start)) s"
done
