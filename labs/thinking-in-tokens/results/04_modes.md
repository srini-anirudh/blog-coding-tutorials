# hardware: NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2, transformers 5.22.2
# command: python labs/thinking-in-tokens/scripts/04_qwen3_runs.py --part modes vote
# model: Qwen/Qwen3-0.6B, 100 GSM8K test questions, first 12 generated tokens, temperature 0.6

| control | thinking off | count |
|---|---|---|
| think | 0.0% | 0/100 |
| soft_think | 0.0% | 0/100 |
| soft_no_think | 100.0% | 100/100 |
| hard_no_think | 100.0% | 100/100 |

think = default template; soft_* append the flag to the user turn; hard_no_think = enable_thinking=False,
which prefills '<think>\n\n</think>\n\n' so the model starts after the closed block.
