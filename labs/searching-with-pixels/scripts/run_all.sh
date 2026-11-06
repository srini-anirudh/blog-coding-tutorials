#!/bin/bash
# Regenerate everything in results/. Set PY to choose the Python. Run from the repository root.
# Script 05 downloads openai/clip-vit-base-patch32 (about 600 MB); pass --skip-clip to it to stay offline.
set -euo pipefail
PY=${PY:-python}
S=labs/searching-with-pixels/scripts
$PY $S/01_metrics.py
$PY $S/02_three_scorers.py          # trains bi16, bi64, bi256, late16, cross; checkpoints in runs/
$PY $S/02b_bi_longer.py            # bi256, four times longer
$PY $S/03_cascade.py
$PY $S/04_negatives.py
$PY $S/05_modality_gap.py
$PY $S/06_index_cost.py
