# hardware: NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2, transformers 5.22.2
# command: python labs/thinking-in-tokens/scripts/04_qwen3_runs.py --part budget waits
# model: Qwen/Qwen3-0.6B, 100 GSM8K test questions, thinking mode, cap 2048 thinking tokens, temperature 0.6

| Waits forced | accuracy | mean thinking tokens | Waits used | hit the cap | right now, wrong at 0 / wrong now, right at 0 |
|---|---|---|---|---|---|
| 0 | 0.790 | 1104 | 0.00 | 16% | - |
| 1 | 0.810 | 1312 | 0.74 | 29% | 6 / 4 |
| 2 | 0.750 | 1666 | 1.49 | 55% | 5 / 9 |

Each run samples afresh, so some flips between rows are sampling noise rather than the Wait.
