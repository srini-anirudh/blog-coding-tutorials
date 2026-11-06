#!/usr/bin/env bash
# Regenerates every file in results/. Run from the repository root. Set PY to choose the Python.
# offline: scripts 01-05 (about 2 minutes on CPU). network: 06-07 download DINOv2 and 1,000 mini-ImageNet images.
set -euo pipefail
PY=${PY:-python}
LAB=labs/inside-a-modern-vision-encoder
what=${1:-all}
if [[ $what == offline || $what == all ]]; then
  $PY $LAB/scripts/01_intake_bench.py
  $PY $LAB/scripts/02_preprocessing.py
  $PY $LAB/scripts/03_pixel_shuffle.py
  $PY $LAB/scripts/04_contrastive.py
  $PY $LAB/scripts/05_patch_n_pack.py
fi
if [[ $what == network || $what == all ]]; then
  $PY $LAB/scripts/06_registers.py
  $PY $LAB/scripts/07_preprocessing_tax.py
fi
