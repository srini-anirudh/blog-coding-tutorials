#!/usr/bin/env bash
# Runs every script in the order of the post. From the repository root: bash labs/across-the-cnnverse/scripts/run_all.sh
# CPU parts take about 2 minutes on 8 threads. 06_degradation needs a GPU for the full run (about 10 minutes per depth).
set -eu
PY=${PY:-python}
L=labs/across-the-cnnverse/scripts
for s in 01_conv_from_scratch 02_receptive_fields 03_equivariance 04_architecture_zoo 05_residual_gradients; do
  "$PY" "$L/$s.py"
done
"$PY" "$L/07_time_and_space.py" --reps 10
if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  "$PY" "$L/06_degradation.py" --epochs 20 --depths 20 --tag _depth20_cuda
  "$PY" "$L/06_degradation.py" --epochs 20 --depths 56 --tag _depth56_cuda
  "$PY" "$L/07_time_and_space.py" --device cuda --reps 20
else
  echo "no CUDA: running 06_degradation on a 2,000-image subset for 3 epochs instead"
  "$PY" "$L/06_degradation.py" --epochs 3 --depths 20 --subset 2000 --tag _cpu_quick
fi
