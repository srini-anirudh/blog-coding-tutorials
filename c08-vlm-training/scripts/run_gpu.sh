#!/usr/bin/env bash
# GPU runs behind results/gpu/. One GPU per job: choose it with CUDA_VISIBLE_DEVICES.
#   run_gpu.sh <job>     one job: stage1_align, stage2_main, eval_stage2_main, stage2_noreplay, a comparison name, benches
#   run_gpu.sh                    every job in order, on one GPU
# Order: stage1_align first, because every other run starts from its projector. eval_stage2_main needs stage2_main.
# The trainer has no mixed precision, so every run is float32. Results land in c08-vlm-training/results/gpu/.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the assigned GPU}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-../.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"   # a GPU job needs few CPU threads; the data pipeline is single-threaded
mkdir -p results/gpu/logs
G="results_dir=results/gpu out_dir=outputs/gpu"
S1=configs/stage1_align.yaml
S2=configs/stage2_mixture.yaml
INIT="init_from=outputs/gpu/stage1_align/trainable.safetensors"
STEPS_S1=${STEPS_S1:-3000}
STEPS_S2=${STEPS_S2:-1000}   # 8000 steps did not fit the 2-hour budget on 4 GPUs
STEPS_CMP=${STEPS_CMP:-250}
STEPS_LONG=${STEPS_LONG:-1500}   # grounding only: is it undertrained at STEPS_CMP?
LONG="$INIT steps=$STEPS_LONG batch_size=16 warmup=100 eval_n=128 eval_every=0 replay_ratio=0 ppl_passages=0 tasks=[grounding]"
FULL="$INIT steps=$STEPS_S2 batch_size=16 warmup=200 eval_n=128 eval_every=$((STEPS_S2 / 2)) eval_n_mid=48 ppl_passages=128"
COMPARE="$INIT steps=$STEPS_CMP batch_size=16 warmup=100 eval_n=128 eval_every=0 replay_ratio=0 ppl_passages=0"
MANUAL="manual_weights={caption: 1, vqa: 1, ocr: 2, doc: 3, chart: 2, grounding: 3}"

t() {  # t <name> <config> [overrides...]
  local name=$1 cfg=$2
  shift 2
  "$PY" -u scripts/train.py "$cfg" name="$name" $G "$@" 2>&1 | grep --line-buffered -avE "Loading weights|Generating .* split" | tee "results/gpu/logs/$name.log"
}

job() {
  case "$1" in
    # Measures: Stage 1 alignment at GPU scale, caption exact match and step time. Result: results/gpu/stage1_align.json
    stage1_align) t stage1_align $S1 steps=$STEPS_S1 batch_size=32 warmup=100 eval_n=256 ;;
    # Measures: the full Stage 2 six-task recipe with 10% text replay, per-task metrics, text perplexity before and after. Result: results/gpu/stage2_main.json
    stage2_main) t stage2_main $S2 $FULL save=true ;;
    # Measures: the same run without replay, the language forgetting comparison at scale. Result: results/gpu/stage2_noreplay.json
    stage2_noreplay) t stage2_noreplay $S2 $FULL replay_ratio=0 ;;
    # Measures: the Stage 2 checkpoint with real images, grey images and the majority baseline, n=256. Result: results/eval_gpu_stage2_main.json
    eval_stage2_main) "$PY" -u scripts/evaluate.py outputs/gpu/stage2_main/trainable.safetensors --name gpu_stage2_main --n 128 2>&1 \
                        | grep -avE "Loading weights" | tee results/gpu/logs/eval_stage2_main.log ;;
    # Measures (comparisons, all $STEPS_CMP steps x 16): freezing, mixtures, grounding formats, resolution, LoRA placement. Result: results/gpu/<name>.json
    freeze_lm) t freeze_lm $S2 $COMPARE ;;
    freeze_partial) t freeze_partial $S2 $COMPARE freeze=partial vision_layers_to_train=4 lrs.vision=1e-5 ;;
    freeze_full) t freeze_full $S2 $COMPARE freeze=full ;;
    mix_proportional) t mix_proportional $S2 $COMPARE mixture=proportional ;;
    mix_uniform) t mix_uniform $S2 $COMPARE mixture=uniform ;;
    mix_manual) t mix_manual $S2 $COMPARE mixture=manual "$MANUAL" ;;
    ground_tokens) t ground_tokens $S2 $COMPARE "tasks=[grounding]" ;;
    ground_text) t ground_text $S2 $COMPARE "tasks=[grounding]" grounding_format=text ;;
    ground_tokens_spread) t ground_tokens_spread $S2 $COMPARE "tasks=[grounding]" loc_init=spread ;;
    # Measures: the three grounding variants again at STEPS_LONG steps. Results: results/gpu/ground_{tokens,tokens_spread,text}_long.json
    ground_tokens_long) t ground_tokens_long $S2 $LONG ;;
    ground_tokens_spread_long) t ground_tokens_spread_long $S2 $LONG loc_init=spread ;;
    ground_text_long) t ground_text_long $S2 $LONG grounding_format=text ;;
    res_tile) t res_tile $S2 $COMPARE "tasks=[doc,ocr]" ;;
    res_resize) t res_resize $S2 $COMPARE "tasks=[doc,ocr]" strategy=resize ;;
    lora_lm|lora_vision|lora_projector) t "$1" $S2 $COMPARE freeze=none "lora=[${1#lora_}]" train_new_tokens=true ;;
    lora_all) t lora_all $S2 $COMPARE freeze=none "lora=[lm,vision,projector]" train_new_tokens=true ;;
    # Measures: padded and packed forward + backward throughput and peak GPU memory. Result: results/gpu/bench_packing.json
    bench_packing) "$PY" benchmarks/bench_packing.py --out gpu/bench_packing.json 2>&1 | tee results/gpu/logs/bench_packing.txt ;;
    # Measures: vision encoder time per document by resolution strategy. Result: results/gpu/bench_resolution.json
    bench_resolution) "$PY" benchmarks/bench_resolution.py gpu/bench_resolution.json 2>&1 | tee results/gpu/logs/bench_resolution.txt ;;
    *) echo "unknown job $1" >&2; exit 2 ;;
  esac
}

ALL="stage1_align stage2_main eval_stage2_main stage2_noreplay freeze_lm freeze_partial freeze_full mix_proportional mix_uniform
     mix_manual ground_tokens ground_text ground_tokens_spread ground_tokens_long ground_tokens_spread_long ground_text_long res_tile res_resize lora_lm lora_vision lora_projector lora_all
     bench_packing bench_resolution"
if [ $# -gt 0 ]; then for j in "$@"; do job "$j"; done; else for j in $ALL; do job "$j"; done; fi
