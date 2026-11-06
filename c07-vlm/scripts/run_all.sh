#!/usr/bin/env bash
# Regenerate everything in results/. Run it from anywhere.
#
#   bash c07-vlm/scripts/run_all.sh          # both parts below
#   bash c07-vlm/scripts/run_all.sh steps    # step scripts and benchmarks
#   bash c07-vlm/scripts/run_all.sh align    # four alignment runs, one after another, then the evaluation
#   PY=path/to/python OMP_NUM_THREADS=4 bash c07-vlm/scripts/run_all.sh
#
# The README gives the time each part took and on what hardware.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

steps() {
    mkdir -p results/steps
    for script in step01_image_processor step02_vision_features step03_projector step04_scatter step05_sequence \
                  step07_positions_masks step09_variable_resolution step10_multi_image step11_generate \
                  step12_save_load break_it; do
        echo "== $script"
        "$PY" "scripts/$script.py" | tee "results/steps/$script.txt"
    done
    echo "== benchmarks"
    "$PY" benchmarks/bench_processor.py | tee results/bench_processor.txt
    "$PY" benchmarks/bench_forward.py --device cpu | tee results/bench_forward_cpu.txt
    "$PY" benchmarks/bench_decode.py | tee results/bench_decode_cpu.txt
}

align() {
    "$PY" scripts/train_alignment.py --projector mlp --steps 300 --eval-every 25
    "$PY" scripts/evaluate_alignment.py --run align-mlp-s2
    "$PY" scripts/train_alignment.py --projector linear --steps 300 --eval-every 50
    "$PY" scripts/train_alignment.py --projector mlp --bad-labels --steps 300 --eval-every 50
    # Four times the image tokens, so each step costs more: fewer steps keep the run near 30 minutes on CPU.
    "$PY" scripts/train_alignment.py --projector mlp --pixel-shuffle 1 --steps 200 --eval-every 50
}

case "${1:-all}" in
    steps) steps ;;
    align) align ;;
    all) steps; align ;;
    *) echo "usage: run_all.sh [steps|align|all]" >&2; exit 2 ;;
esac
