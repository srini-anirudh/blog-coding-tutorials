# hardware: NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2, transformers 5.22.2
# command: python labs/thinking-in-tokens/scripts/04_qwen3_runs.py --part modes vote
# model: Qwen/Qwen3-0.6B, 200 GSM8K test questions, non-thinking mode (hard prefill), 16 samples each, temperature 0.7, top-p 0.8, top-k 20, max 512 tokens; generation took 383 s

| k samples | vote@k (plurality) | pass@k (oracle verifier picks) |
|---|---|---|
| 1 | 0.618 | 0.619 |
| 2 | 0.617 | 0.717 |
| 4 | 0.665 | 0.793 |
| 8 | 0.688 | 0.856 |
| 16 | 0.715 | 0.915 |

mean single-sample accuracy: 0.619
questions where the right answer is the mode of the 16 samples: 140/200
  of those, right in at most half the samples (a minority the vote still recovers): 15
questions where a wrong answer is the mode with at least half the samples (vote confirms an error): 33

vote@k for k < 16 averages 64 random size-k subsets of the 16 samples; pass@k is the unbiased estimator.
