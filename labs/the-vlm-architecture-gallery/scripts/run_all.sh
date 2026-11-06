#!/usr/bin/env bash
# Regenerate everything in results/. Run from the repository root. Set PY to choose the Python.
# 01-04 take seconds; 05 and 06 train tiny models (about 10 to 20 minutes each on a shared 8-thread CPU);
# 07 needs network access to the Hugging Face Hub.
set -euo pipefail
PY=${PY:-python}
D=labs/the-vlm-architecture-gallery/scripts
$PY $D/01_connectors.py
$PY $D/02_input_tokens.py
$PY $D/03_fusion_identity.py
$PY $D/04_kv_cost.py
for s in 0 1; do $PY $D/05_connector_bench.py --seed $s; done
$PY $D/05_connector_bench.py --report
for s in 0 1; do $PY $D/06_fusion_freeze.py --seed $s; done
$PY $D/06_fusion_freeze.py --report
$PY $D/07_hub_configs.py
$PY $D/08_xattn_longer.py
