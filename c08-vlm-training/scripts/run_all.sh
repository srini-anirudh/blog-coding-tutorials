#!/usr/bin/env bash
# Every training run behind the numbers in the README and the post.
# Usage: scripts/run_all.sh [stage1|group_a|group_b]   (default: all, in order)
# On a shared laptop, set VLMTRAIN_DEVICE=cpu to avoid waiting on a busy GPU queue.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-../.venv/bin/python}"
S2=configs/stage2_mixture.yaml
mkdir -p results/logs

t() {  # t <name> <config> [overrides...]
  local name=$1 cfg=$2
  shift 2
  [ -f "results/$name.json" ] && { echo "skip $name"; return; }
  "$PY" -u scripts/train.py "$cfg" name="$name" "$@" 2>&1 | grep --line-buffered -avE "Loading weights|UNEXPECTED|LOAD REPORT|^Key  |^---|^Notes|can be ignored|Redirects|unauthenticated" > "results/logs/$name.log"
}

stage1() { "$PY" -u scripts/train.py configs/stage1_align.yaml 2>&1 | grep --line-buffered -avE "Loading weights|UNEXPECTED|LOAD REPORT|^Key  |^---|^Notes|can be ignored|Redirects|unauthenticated" > results/logs/stage1_align.log; }

group_a() {
  t stage2_main "$S2"                                        # T=2 mixture, 10% replay
  t stage2_noreplay "$S2" replay_ratio=0                     # the forgetting comparison
  t stage2_partial "$S2" steps=150 eval_every=0 freeze=partial vision_layers_to_train=2 lrs.vision=1e-5
  t stage2_lora "$S2" steps=150 eval_every=0 freeze=none "lora=[lm,projector]" train_new_tokens=true
}

group_b() {
  t mix_proportional "$S2" steps=150 eval_every=0 mixture=proportional replay_ratio=0 ppl_passages=0
  t mix_uniform "$S2" steps=150 eval_every=0 mixture=uniform replay_ratio=0 ppl_passages=0
  t ground_tokens "$S2" steps=150 eval_every=0 "tasks=[grounding]" replay_ratio=0 ppl_passages=0 eval_n=48
  t ground_text "$S2" steps=150 eval_every=0 "tasks=[grounding]" grounding_format=text replay_ratio=0 ppl_passages=0 eval_n=48
  t res_tile "$S2" steps=150 eval_every=0 "tasks=[doc,ocr]" replay_ratio=0 ppl_passages=0 eval_n=48
  t res_resize "$S2" steps=150 eval_every=0 "tasks=[doc,ocr]" strategy=resize replay_ratio=0 ppl_passages=0 eval_n=48
  t stage2_replay_tokenmean "$S2" loss_norm=token eval_every=0   # 10% of samples, most of the tokens
}

case "${1:-all}" in
  stage1) stage1 ;;
  group_a) group_a ;;
  group_b) group_b ;;
  all) stage1; group_a; group_b ;;
esac
