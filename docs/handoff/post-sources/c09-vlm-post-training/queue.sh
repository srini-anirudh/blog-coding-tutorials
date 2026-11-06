#!/usr/bin/env bash
# Remaining steps of scripts/run_all.sh, skipping what already finished.
cd <CODE>/c09-vlm-post-training
PY=../.venv/bin/python
run() { name=$1; shift; echo "=== $name" >> results/run_all.log; "$PY" -u "$@" > "results/$name.log" 2>&1; echo "exit $? $name" >> results/run_all.log; }
run dpo_balanced      scripts/train_dpo.py --negatives balanced --steps 150
run dpo_biased        scripts/train_dpo.py --negatives biased --steps 150
run grpo_llm          scripts/train_grpo.py --freeze llm --steps 60 --eval both
run grpo_projector    scripts/train_grpo.py --freeze projector+llm --steps 60
run grpo_all          scripts/train_grpo.py --freeze all --steps 60
run perception_probe  scripts/perception_probe.py --runs grpo_count_exact_llm grpo_count_exact_projector_llm grpo_count_exact_all
run hack_recall       scripts/train_grpo.py --task caption --reward color_recall --steps 60 --max-new-tokens 24 --eval both
run hack_f1           scripts/train_grpo.py --task caption --reward color_f1 --steps 60 --max-new-tokens 24
run bench_encode_once benchmarks/bench_encode_once.py --groups 4 8 16
