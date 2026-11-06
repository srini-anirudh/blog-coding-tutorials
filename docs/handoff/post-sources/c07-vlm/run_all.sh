#!/bin/zsh
cd <CODE>
export HF_HUB_OFFLINE=1
L=.series/code-series/c07-vlm/logs
.venv/bin/python c07-vlm/scripts/train_alignment.py --projector mlp --steps 200 --eval-every 25 > $L/align-mlp.log 2>&1
.venv/bin/python c07-vlm/scripts/evaluate_alignment.py --run align-mlp-s2 > $L/eval-mlp.log 2>&1
.venv/bin/python c07-vlm/scripts/train_alignment.py --projector linear --steps 200 --eval-every 50 > $L/align-linear.log 2>&1
.venv/bin/python c07-vlm/scripts/train_alignment.py --projector mlp --bad-labels --steps 200 --eval-every 50 > $L/align-bad.log 2>&1
.venv/bin/python c07-vlm/scripts/train_alignment.py --projector mlp --pixel-shuffle 1 --steps 200 --eval-every 50 > $L/align-s1.log 2>&1
echo ALLDONE > $L/all.done
