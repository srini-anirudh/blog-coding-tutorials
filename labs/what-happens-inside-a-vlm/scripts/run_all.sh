#!/bin/bash
# Run every experiment in order. Each script writes results/<script>.txt and .json.
# Usage, from the repository root: bash labs/what-happens-inside-a-vlm/scripts/run_all.sh [extra args, e.g. --n 2]
set -uo pipefail  # keep going if one script fails, so one bug does not hide the others
PY=${PY:-python}
D=$(dirname "$0")
for s in 01_blind_baseline 02_logit_lens 03_attention_sinks_ablation 04_knockout 05_patching \
         06_probe_vs_readout 07_connector; do
  echo "=== $s"
  /usr/bin/time -f "$s took %e s" "$PY" "$D/$s.py" "$@"
done
