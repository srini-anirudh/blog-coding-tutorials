#!/usr/bin/env bash
# Every real run of this lab, in order. Run from the repository root.
# The chart training takes 20 to 25 minutes per seed at 3 threads on a shared machine.
set -euo pipefail
S=labs/rewarding-what-the-model-sees/scripts
python $S/audit_rewards.py
python $S/advantage_math.py
python $S/credit_assignment.py
python $S/train_box_policy.py --steps 300 --seeds 0 1 2
# one seed trains six configs (the three experiments share them); seeds are independent, so run them in parallel
for s in 0 1 2; do python $S/train_chart_policy.py --seeds $s --train-only --threads 3 & done
wait
python $S/train_chart_policy.py --seeds 0 1 2 --report-only
