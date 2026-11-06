#!/bin/zsh
# runs the experiments with up to 3 at a time, each logs to results/<name>.log
C=<CODE>/c12-agent-rl; cd $C
PY=../.venv/bin/python
( $PY scripts/demo_env.py 2>&1 | grep -v Redirects > results/demo_env.txt
  $PY scripts/distribution_shift.py 2>&1 | grep -v Redirects > results/shift_bc.txt
  $PY benchmarks/bench_rollout.py 2>&1 | grep -v Redirects > results/bench_rollout.txt ) &
( $PY -u scripts/experiments.py main 2>&1 | grep -v Redirects > results/main.log; $PY scripts/recovery_demo.py 2>&1 | grep -v Redirects > results/recovery.log; $PY -u scripts/experiments.py exploration 2>&1 | grep -v Redirects > results/exploration.log ) &
( $PY -u scripts/experiments.py shaping 2>&1 | grep -v Redirects > results/shaping.log; $PY -u scripts/experiments.py credit 2>&1 | grep -v Redirects > results/credit.log; $PY scripts/export_figures.py 2>&1 | grep -v Redirects > results/export_figures.log; $PY -u scripts/train_system.py --iters 10 2>&1 | grep -v Redirects > results/train_system.txt ) &
( $PY -u scripts/experiments.py curriculum 2>&1 | grep -v Redirects > results/curriculum.log; $PY -u scripts/experiments.py offline 2>&1 | grep -v Redirects > results/offline.log; $PY -u scripts/experiments.py groupsize 2>&1 | grep -v Redirects > results/groupsize.log ) &
wait
echo ALL_DONE
