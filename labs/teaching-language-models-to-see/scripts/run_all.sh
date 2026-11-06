#!/usr/bin/env bash
# Regenerate every file in results/. Run from the repository root. Set PY to choose the Python.
# s6 and s7 load the connector that s2 trains, so s2 runs first.
set -euo pipefail
PY=${PY:-python}
L=labs/teaching-language-models-to-see/scripts
$PY $L/s1_contrastive.py
$PY $L/s2_connector_gap.py
$PY $L/s3_gated_xattn.py
$PY $L/s4_freeze_cost.py --batch-size 2 --repeats 5
$PY $L/s5_token_budget.py
$PY $L/s6_learning_signal.py
$PY $L/s7_alignment_stage.py
# On a GPU: $PY $L/s4_freeze_cost.py --device cuda --batch-size 8 --repeats 10
