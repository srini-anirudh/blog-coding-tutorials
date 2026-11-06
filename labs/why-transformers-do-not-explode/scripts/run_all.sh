#!/usr/bin/env bash
# Run every script in order from the repository root. About 1 hour on 8 CPU threads of a quiet machine;
# 04 and 06 (the training sweeps) take most of it. Scripts 04-09 download Tiny Shakespeare (1.1 MB) if needed.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
for s in 01_variance_and_init 02_signal_propagation 03_residual_stream 05_deepnorm_update 07_adam_under_stress \
         08_number_formats; do
  $PY labs/why-transformers-do-not-explode/scripts/$s.py
done
$PY labs/why-transformers-do-not-explode/scripts/04_norm_placement.py --steps 400 --warmup 100
$PY labs/why-transformers-do-not-explode/scripts/06_attention_logits.py
$PY labs/why-transformers-do-not-explode/scripts/09_mup_transfer.py
