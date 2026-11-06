#!/usr/bin/env bash
# Every training run behind the numbers in the README and the post, one at a time.
# Usage: scripts/run_all.sh [stage1|stage2|freeze|mixture|grounding|resolution|lora|packing|own|capture]   (default: all, in order)
# A finished run is skipped. Delete results/<name>.json to run it again.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-../.venv/bin/python}"
S2=configs/stage2_mixture.yaml
mkdir -p results/logs

# Download progress and model-loading reports say nothing about training, so they stay out of the logs.
quiet() { grep --line-buffered -avE "Loading weights|UNEXPECTED|LOAD REPORT|^Key  |^---|^Notes|can be ignored|Redirects|unauthenticated|Generating .* split"; }

t() {  # t <name> <config> [overrides...]
  local name=$1 cfg=$2
  shift 2
  [ -f "results/$name.json" ] && { echo "skip $name"; return; }
  echo "run $name"
  "$PY" -u scripts/train.py "$cfg" name="$name" "$@" 2>&1 | quiet > "results/logs/$name.log"
}

SHORT="steps=200 eval_every=0 replay_ratio=0 ppl_passages=0"

stage1() { t stage1_align configs/stage1_align.yaml; }

stage2() {
  t stage2_main "$S2" save=true                              # T=2 mixture, 10% replay
  t stage2_noreplay "$S2"  replay_ratio=0                    # the forgetting comparison
  t stage2_replay_tokenmean "$S2" steps=100 eval_every=0 loss_norm=token   # 10% of samples, most of the tokens
}

# The same 200 steps with more of the model unfrozen. mix_temperature doubles as freeze=lm.
freeze() {
  t freeze_partial "$S2" $SHORT freeze=partial vision_layers_to_train=2 lrs.vision=1e-5
  t freeze_full "$S2" $SHORT freeze=full lrs.vision=1e-6
}

mixture() {
  t mix_temperature "$S2" $SHORT mixture=temperature
  t mix_proportional "$S2" $SHORT mixture=proportional
  t mix_uniform "$S2" $SHORT mixture=uniform
  t mix_manual "$S2" $SHORT mixture=manual \
    "manual_weights={caption: 1, vqa: 1, ocr: 2, doc: 3, chart: 2, grounding: 3}"   # more of what is hard
}

grounding() {
  t ground_tokens "$S2" $SHORT "tasks=[grounding]"
  t ground_text "$S2" $SHORT "tasks=[grounding]" grounding_format=text
  t ground_tokens_spread "$S2" $SHORT "tasks=[grounding]" loc_init=spread   # distinct <loc_k> rows from step 0
}

resolution() {
  t res_tile "$S2" $SHORT "tasks=[doc,ocr]"
  t res_resize "$S2" $SHORT "tasks=[doc,ocr]" strategy=resize
}

lora() {  # train_new_tokens: the <loc_k> rows must learn even though the LM itself is frozen
  t lora_lm "$S2" $SHORT steps=150 freeze=none "lora=[lm]" train_new_tokens=true
  t lora_vision "$S2" $SHORT steps=150 freeze=none "lora=[vision]" train_new_tokens=true
  t lora_projector "$S2" $SHORT steps=150 freeze=none "lora=[projector]" train_new_tokens=true
  t lora_all "$S2" $SHORT steps=150 freeze=none "lora=[lm,vision,projector]" train_new_tokens=true
}

packing() {  # throughput only, so the runs are short and the eval is token
  t train_packed "$S2" $SHORT steps=60 batch_size=8 eval_n=8 pack_budget=512
  t train_padded "$S2" $SHORT steps=60 batch_size=8 eval_n=8
}

# The fallback model in vlmtrain/model.py, on real weights, so "this folder runs alone" is a tested claim.
own() { t own_smoke configs/stage1_align.yaml model=own steps=20 batch_size=4 warmup=2 log_every=5 eval_n=8 save=false; }

# No training here, apart from the short Break it runs.
capture() {
  local log=results/logs
  "$PY" scripts/show_samples.py 2>&1 | quiet > $log/show_samples.txt
  "$PY" scripts/inspect_batch.py 2>&1 | quiet > $log/inspect_batch.txt
  "$PY" scripts/param_report.py 2>&1 | quiet > $log/param_report.txt
  "$PY" scripts/multi_image.py 2>&1 | quiet > $log/multi_image.txt
  "$PY" benchmarks/bench_resolution.py 2>&1 | quiet > $log/bench_resolution.txt
  "$PY" benchmarks/bench_packing.py 2>&1 | quiet > $log/bench_packing.txt
  "$PY" scripts/evaluate.py outputs/stage1_align/trainable.safetensors --name stage1_align --tasks caption 2>&1 | quiet > $log/eval_stage1_align.txt
  "$PY" scripts/evaluate.py outputs/stage2_main/trainable.safetensors --name stage2_main 2>&1 | quiet > $log/eval_stage2_main.txt
  for b in image_tokens loss_mask block_mask max_length high_lr frozen_loc; do
    "$PY" scripts/break_it.py $b 2>&1 | quiet > $log/break_$b.txt
  done
}

case "${1:-all}" in
  stage1|stage2|freeze|mixture|grounding|resolution|lora|packing|own|capture) "$1" ;;
  all) stage1; stage2; mixture; freeze; grounding; resolution; lora; packing; own; capture ;;
  *) echo "unknown group $1" >&2; exit 2 ;;
esac
