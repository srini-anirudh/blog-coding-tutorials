#!/usr/bin/env bash
# Regenerate everything in results/. Run from the repository root. Set PY to choose the Python.
# cpu: scripts that run on a CPU in a few minutes. gpu: the two Qwen2-VL experiments on one CUDA GPU.
set -euo pipefail
PY=${PY:-python}
L=labs/the-cost-of-a-picture/scripts
what=${1:-cpu}
if [[ $what == cpu || $what == all ]]; then
  $PY $L/01_cost_model.py
  $PY $L/02_connectors.py
  $PY $L/03_stage_timing.py --device cpu --sizes 448x448,672x672 --decode-steps 16 --repeats 2
  $PY $L/04_pruning.py --device cpu --n 16
  $PY $L/05_prefix_cache.py --device cpu
fi
if [[ $what == gpu || $what == all ]]; then
  $PY $L/03_stage_timing.py --device cuda --eager --repeats 5
  $PY $L/04_pruning.py --device cuda --n 48
fi
