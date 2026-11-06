# GPU run still to do for this lab. Run from the repository root on one CUDA GPU.
# The pruning run (04_pruning.py --device cuda --n 48) is done; its output is results/04_pruning_cuda.txt.
# Measures preprocess / encoder / prefill / decode-step time and KV bytes of Qwen2-VL-2B (BF16) at four image sizes,
# plus prefill with eager attention against SDPA. Writes results/03_stage_timing_cuda.{txt,json}.
python labs/the-cost-of-a-picture/scripts/03_stage_timing.py --device cuda --eager --repeats 5
