#!/usr/bin/env bash
# Every real run behind the post, one after another. Logs and JSON land in results/.
# About four hours on 8 threads of an AMD EPYC 7R13 (see the README for the run times).
set -u
cd "$(dirname "$0")/.."
PY="${PY:-../.venv/bin/python}"
export TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error  # keep progress bars out of the logs
exec > >(tee results/run_all.log) 2>&1
run() { name=$1; shift; start=$SECONDS; echo "=== $name"; "$PY" -u "$@" > "results/$name.log" 2>&1; echo "exit $? $name $((SECONDS - start))s"; }

run show_pairs        scripts/show_pairs.py
run blind_probe       scripts/blind_probe.py
run reward_audit      scripts/reward_audit.py
run show_rollouts     scripts/show_rollouts.py
run bench_encode_once benchmarks/bench_encode_once.py --groups 4 8 16
run dpo_balanced      scripts/train_dpo.py --negatives balanced --steps 150
run dpo_biased        scripts/train_dpo.py --negatives biased --steps 150
run grpo_llm          scripts/train_grpo.py --freeze llm --steps 60 --eval both
run grpo_projector    scripts/train_grpo.py --freeze projector+llm --steps 60
run grpo_all          scripts/train_grpo.py --freeze all --steps 60
run perception_probe  scripts/perception_probe.py --runs grpo_count_exact_llm grpo_count_exact_projector_llm grpo_count_exact_all
run hack_recall       scripts/train_grpo.py --task caption --reward color_recall --steps 60 --max-new-tokens 24 --eval both
run hack_f1           scripts/train_grpo.py --task caption --reward color_f1 --steps 60 --max-new-tokens 24
