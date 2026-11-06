# hardware: NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2, transformers 5.22.2
# command: python labs/thinking-in-tokens/scripts/04_qwen3_runs.py --part budget waits
# model: Qwen/Qwen3-0.6B, 100 GSM8K test questions, thinking mode, temperature 0.6, top-p 0.95, top-k 20, answer phase up to 512 tokens

| thinking budget | accuracy | mean thinking tokens used | cut off by the budget | seconds |
|---|---|---|---|---|
| 0 | 0.610 | 0 | 0% | 49 |
| 64 | 0.680 | 64 | 100% | 28 |
| 128 | 0.590 | 128 | 100% | 29 |
| 256 | 0.690 | 256 | 100% | 38 |
| 512 | 0.770 | 497 | 85% | 59 |
| 1024 | 0.810 | 823 | 38% | 109 |
| 2048 | 0.780 | 1131 | 23% | 226 |

Budget 0 is the hard switch (empty think block prefilled). At the cap the controller splices in Qwen3's
stop-thinking sentence and </think>, then the model answers from the partial trace.
