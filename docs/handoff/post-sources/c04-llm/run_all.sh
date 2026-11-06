#!/bin/zsh
# Sequential real runs for C4. Each writes straight to c04-llm/results/.
cd <CODE>
PY=.venv/bin/python
R=c04-llm/results
run() { out=$1; shift; "$@" > "$R/$out" 2> >(grep -v "Redirects\|Loading weights" >&2); echo "done $out $(date +%T)"; }
run train_sanity.txt $PY -u c04-llm/scripts/train_sanity.py
run generate_compare.txt $PY c04-llm/scripts/generate.py --compare
run walkthrough.txt $PY c04-llm/scripts/walkthrough.py --step all
run inspect_shapes.txt $PY c04-llm/scripts/inspect_shapes.py
run param_count.txt $PY c04-llm/scripts/param_count.py
run bench_generation_mps.txt $PY c04-llm/benchmarks/bench_generation.py --device mps
run bench_generation_cpu.txt $PY c04-llm/benchmarks/bench_generation.py --device cpu
run bench_generation_smollm2_cpu.txt $PY c04-llm/benchmarks/bench_generation.py --device cpu --preset smollm2-135m --new-tokens 128
run bench_attention_cpu.txt $PY c04-llm/benchmarks/bench_attention.py --device cpu
run bench_attention_mps.txt $PY c04-llm/benchmarks/bench_attention.py --device mps
run compare_hf.txt $PY c04-llm/scripts/compare_hf.py --pretrained HuggingFaceTB/SmolLM2-135M
run break_it.txt $PY c04-llm/scripts/break_it.py --case all --pretrained HuggingFaceTB/SmolLM2-135M --device cpu
run moe_demo.txt $PY -u c04-llm/scripts/moe_demo.py --steps 300
run compare_heads.txt $PY -u c04-llm/scripts/compare_heads.py --steps 400
echo ALLDONE
