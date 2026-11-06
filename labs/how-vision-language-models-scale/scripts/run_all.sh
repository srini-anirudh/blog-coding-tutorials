#!/bin/bash
# Regenerate everything in results/. Run from the repository root. Set PY to choose the Python.
set -euo pipefail
PY=${PY:-python}
D=labs/how-vision-language-models-scale/scripts
$PY $D/01_budget.py
$PY $D/02_measure_k.py
$PY $D/03_compute_ladder.py
$PY $D/04_quality_quantity.py        # about 12 min on a busy CPU
$PY $D/05_laws.py
$PY $D/06_tokens_vs_size.py          # about 30 min on a busy CPU, 6 parallel jobs
if $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  $PY $D/07_frozen_step_gpu.py       # needs CUDA, about 1 min on one H100
fi
