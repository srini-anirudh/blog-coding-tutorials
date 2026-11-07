# Lab: The Mechanics of LLM Inference

Code companion for the post [The Mechanics of LLM Inference](https://srini-anirudh.github.io/blogs/the-mechanics-of-llm-inference/).

The post derives inference cost from bytes and FLOPs. This lab does two things with those derivations.
First, it recomputes every paper number in the post from one small formula module, and flags the ones that do not match.
Second, it measures the main claims on one H100. It uses a Llama-3-8B-shaped decoder with random weights.
The speed of a step does not depend on what the weights are, only on their shape and format.

What you can watch happen:

- The KV cache is exact. Earlier keys and values are bit-identical after a token is appended. Cached and uncached greedy generation give the same tokens.
- One weight matrix behaves as two machines. At 1 row it is bandwidth-bound, and at 8,192 rows it is compute-bound.
- Decode steps sit on the floor that bytes / bandwidth sets, and the floor rises as the context grows.
- Batching shares the weight read but not the caches. Full MHA runs out of memory where GQA does not.
- Halving the bytes per weight cuts decode time. It does not speed up prefill.
- Verifying several speculative tokens costs about one step. Real speculative decoding with a SmolLM2 pair reproduces the target's greedy output exactly.
- Static batching wastes most slots on the post's eight requests. Continuous batching recovers them.

## Layout

```text
labs/the-mechanics-of-llm-inference/
├── inferlab/
│   ├── ledger.py        every formula in the post: KV bytes, prefill/decode work, decode floor, B*, I(B), speculation, MoE, TP
│   ├── decoder.py       Llama-shaped decoder with a preallocated KV cache (HF weight names; loads SmolLM2)
│   ├── speculative.py   speculative decoding with the rejection rule, plus a plain-decoding baseline
│   ├── scheduling.py    static against continuous batching, in decode-step units
│   └── timing.py        CUDA-event timing, CUDA-graph capture, measured bandwidth and peak, results headers
├── scripts/             01 to 08, one per post claim; run_gpu.sh runs the GPU ones in order
├── tests/               pytest
└── results/             real outputs; each file starts with its hardware and command
```

The teaching model for this series is C4's `tinyllm` ([c04-llm](../../c04-llm)). It grows its cache with `torch.cat`, so tensor shapes change every step.
`inferlab/decoder.py` keeps the same maths and weight names, but it writes into one fixed buffer.
That buffer is what makes three things possible here:

- a decode step reads exactly the weights plus the cache slice, with no copies;
- a step can be captured as a CUDA graph, which removes Python and launch overhead from the timing;
- rejected speculative tokens are rolled back by moving the write position.

`tests/test_decoder.py` checks it against `tinyllm` and against Hugging Face `LlamaForCausalLM`.
Attention and GQA themselves are built in [c03-attention](../../c03-attention).

## Setup

From the repository root, after `pip install -e ".[dev]"`. Tested with Python 3.12, PyTorch 2.15 (CUDA 13.2), transformers 5.22 and Triton 3.9.
Script 06 needs Triton, which ships with the CUDA build of PyTorch. Script 07 part 3 needs SmolLM2-135M and SmolLM2-360M in the local Hugging Face cache. It skips that part if they are missing.

## Run everything

```bash
L=labs/the-mechanics-of-llm-inference
# CPU, under a minute each
python $L/scripts/01_kv_cache_exact.py --out $L/results/kv_cache_exact_cpu.txt
python $L/scripts/02_ledger.py         --out $L/results/ledger_check.txt
python $L/scripts/08_scheduling.py     --out $L/results/scheduling.txt
# One GPU, about 10 minutes: scripts 03 to 07 and the GPU tests
bash $L/scripts/run_gpu.sh

# Tests
python -m pytest $L -m "not slow and not gpu and not network"   # offline, about 20 s
python -m pytest $L -m network                                  # SmolLM2-135M against Hugging Face
python -m pytest $L -m gpu                                      # CUDA-graph replay equals eager
```

## Results

Hardware: **NVIDIA H100 80GB HBM3**, CUDA 13.2, PyTorch 2.15.0, one GPU, no other job on it (the run takes an exclusive lock). CPU rows: **AMD EPYC 7R13, 8 threads**.
The decoder is Llama-3-8B-shaped with random bf16 weights, so these are timings of a plain PyTorch implementation (SDPA attention, no fused kernels), not of a tuned engine.
Every row comes from a file in `results/`. Each file begins with its hardware and command.

This GPU copies memory at 3.03 TB/s (spec 3.35) and multiplies 8192-square bf16 matrices at 769 TFLOP/s (spec 989). Its measured ridge is 254 FLOP/byte, against the post's 295.

| Post claim | Post's number | Measured here | File |
|---|---|---|---|
| Decode is bandwidth-bound: one 4096x4096 layer | I = 1 FLOP/byte, 0.3% of the roof | 1 row: 1.2 TFLOP/s, 0.1% of spec peak. 28 us, the same for 1 to 16 rows. | `roofline_gemm_h100.txt` |
| Prefill is compute-bound: the same layer | I = 1638 | 8192 rows: 715 TFLOP/s, 72% of spec peak. Time is flat up to 512 rows, then roughly linear. | `roofline_gemm_h100.txt` |
| Prefill of an 8,192-token prompt, 8B model | 149 TFLOP, about 335 ms at 45% of peak | 149 TFLOP in 256 ms: 582 TFLOP/s, 59% of spec peak, 31 us per token | `decode_wall_h100.txt` |
| Single-stream decode step, 8B, 8k context | floor 5.1 ms (spec), 100 to 140 tok/s measured elsewhere | 8.6 ms as a CUDA graph, 117 tok/s. Floor from measured bandwidth: 5.3 ms. Eager PyTorch: 28 ms. | `decode_wall_h100.txt` |
| The wall moves with context | 128k: cache 17.2 GB against 16.1 GB of weights, step more than doubles | Step grows from 8.2 ms (512) to 13.6 ms (128k), 74 tok/s | `decode_wall_h100.txt` |
| Full MHA is four times the cache | Llama-2-7B shape at 128k: 68.7 GB | Step is 29.0 ms at 128k, 34 tok/s, against 13.6 ms for the GQA 8B | `decode_wall_h100.txt` |
| Batching shares the weight read | Flat step, rising throughput | Context 512: B=1 to 64 takes the step from 8.2 to 10.8 ms. Aggregate throughput rises from 122 to 5,948 tok/s. | `batching_h100.txt` |
| Caches are private | B_max = 57 at 8k context | 8k context: B=56 runs and B=64 does not fit. Step 8.6 ms at B=1, 28.4 ms at B=56. Aggregate 1,972 tok/s. | `batching_h100.txt` |
| Verifying gamma + 1 tokens is nearly free below B* | One step's cost | Batch 1: 1.2x for 2 positions and 1.3x for 3 to 64. Batch 8 x 5 rows: 1.5x. Batch 64 x 5 rows (320 rows, past the ridge): 3.4x. | `batching_h100.txt` |
| Weight bytes set decode time | 4x fewer bytes, 4x faster ceiling | One 14336x4096 matrix: fp32 89 us, bf16 44 us, int8 29 us (1.5x faster than bf16, not 2x) | `quantization_h100.txt` |
| Weight-only quantization does not help prefill | Can be a net loss | 4096 rows: bf16 683 us, int8 dequantized then multiplied 972 us (0.70x) | `quantization_h100.txt` |
| KV cache is exact | Bit-identical | Keys and values at earlier positions: max abs difference 0, all 30 layers. Cached and uncached greedy tokens equal. | `kv_cache_exact_cpu.txt` |
| Cache removes recomputation | 417x for the 8B example | 41.8x fewer rows for a 100-token prompt and 50 new tokens. Wall time 1.9x on 8 CPU threads (see the caveat below). | `kv_cache_exact_cpu.txt` |
| Speculation preserves the distribution | Provably identical | 1,000,000 draws: total variation from the target 0.0008 (sampling noise alone is about 0.0014) | `speculative_h100.txt` |
| E[tokens] = (1 - a^(g+1)) / (1 - a) | 3.36 at a=0.8, g=4 | Simulated 3.366. SmolLM2-135M drafting for SmolLM2-360M: a = 0.88, 3.71 tokens per pass at g=4 (formula with that a: 3.95). | `speculative_h100.txt` |
| Static batching wastes slots | 29.5% utilisation, 8 slots | 29.5%. With 4 slots: static 36.9%, continuous 56.7%. With 2 slots: 58.4% and 95.2%. | `scheduling.txt` |
| Paper numbers in the post | | 2 of 77 checks differ from the post (below) | `ledger_check.txt` |

Tests:

| Run | Result |
|---|---|
| Offline: `-m "not slow and not gpu and not network"` | 21 passed in about 16 s |
| Network: SmolLM2-135M logits against Hugging Face | 1 passed |
| GPU: CUDA-graph replay equals eager | 1 passed (`results/pytest_gpu.txt`) |

### What the measurements say

- **The roofline shape is real.** One layer takes 24 to 29 us for every row count from 1 to 512, then climbs. The time starts to grow between 512 and 1024 rows, near the measured ridge of 254 FLOP/byte (the post's 295 on the spec sheet). The M=1 case reaches only 1.2 TB/s here because a 32 MiB matrix is too small to fill the memory system. The larger 14336x4096 matrix in `06_quantization.py` reaches 2.7 TB/s.
- **The plain PyTorch decoder is 60 to 80% above the floor.** The post says real engines land 20 to 50% above. A step takes 8.2 ms at short context against a 5.0 ms floor. The decoder launches many small unfused kernels (RMSNorm in fp32, RoPE, SDPA, SwiGLU), and the step reaches 1.8 TB/s. Eager mode is worse, at 28 ms, because it is launch-bound. The graph replay is what removes that.
- **Verification costs 1.2 to 1.3 steps at batch 1, not 1.0.** The extra rows probably switch the GEMVs to GEMMs and attention to the masked SDPA path. This was not profiled. The cost does not grow from 3 to 64 rows, which is the post's point. At batch 64 with 5 rows the step triples, as the post predicts once B(gamma+1) passes B*.
- **Speculative decoding did not speed up this pair.** Both SmolLM2 models are launch-bound in eager PyTorch, so a draft step costs 0.94 of a target step (c = 0.94). With c that high the formula itself predicts a slowdown (0.83x at gamma 4) and the measurement agrees (0.73x). The acceptance rate (0.88) and the tokens per pass are the part that reproduces. The 2.8x speed-up needs a draft costing about 5% of a target step, and this lab did not measure that.
- **Greedy speculative output is identical to plain greedy output** for all 4 prompts at gamma 2, 4 and 8.
- **The cache's wall-time gain on CPU is small and noisy.** At 135M parameters the CPU spends much of its time outside the linear layers. The row count (41.8x) is exact. The wall-time ratio varied from 1.9x to 3.7x across four runs because the machine is shared.

### Where the post and this lab disagree

- `P / (L d) = 53,000` for the prefill crossover uses 6.98e9 parameters, which excludes the embedding table and the LM head. With the 8.03e9 total, the formula gives 61,000. The post does not say which it used.
- The 2,400-token prefill in the VLM budget was printed as 36.6 TFLOP. A 6.74e9-parameter backbone gives 33.9 TFLOP. The post now uses 6.74e9 parameters, 34 TFLOP, 76 ms and a TTFT of about 112 ms.
- The post says a full-MHA 7B sequence at 128k "needs more than an 80 GB H100 has left". That is true in decimal gigabytes. This card has 85.0 GB of memory, and weights (13.5 GB) plus cache (68.7 GB) fit, so the 128k run above completed.
- The post's 45% of peak for prefill is conservative for this GPU: 59% was measured, with a plain PyTorch implementation.

## Post sections and scripts

| Post section | Script | Runs on |
|---|---|---|
| 3. Prefill and decode, "The shapes are what matter" | `03_roofline_gemm.py` | GPU |
| 3. "Putting numbers on it" | `04_decode_wall.py`, `02_ledger.py` | GPU, CPU |
| 4. Deriving the KV cache | `01_kv_cache_exact.py`, `tests/test_decoder.py` | CPU |
| 5, 7, 8. KV formula, MHA/GQA/MLA, memory ledger | `02_ledger.py`, `04_decode_wall.py` (MHA at 128k) | CPU, GPU |
| 10. Why decode is bandwidth-bound | `04_decode_wall.py` | GPU |
| 11. What batching actually buys | `05_batching.py` | GPU |
| 14. Quantization as a ledger edit | `06_quantization.py` | GPU |
| 15. Speculative decoding | `07_speculative.py`, `inferlab/speculative.py` | CPU, GPU |
| 19. The scheduling problem | `08_scheduling.py` | CPU |

## What was not measured

- Only one GPU was used. Tensor parallelism, pipeline parallelism and expert parallelism (section 16) are computed in `ledger.py` and checked against the post's numbers, but no collective was timed. The 6 us all-reduce latency is the post's figure.
- Vision-language inference (section 17) is arithmetic only. No vision encoder was run. The post's 15 ms preprocessing figure is not checked.
- The decoder is plain PyTorch. A tuned engine (fused kernels, FlashAttention-3, paged attention) would sit closer to the floor. Paged attention, prefix caching and chunked prefill are not implemented. The scheduling simulation is slot accounting in decode-step units, not a timing model, and ignores the cost of prefill interference beyond blocking.
- FP8 weights, activations and KV cache were not run, and quantization quality was not evaluated. The int8 kernel is a simple Triton GEMV with per-output-channel scales on random weights.
- A speculative speed-up was not demonstrated (see above). Acceptance rates come from one pair of small models on four prompts.
- MLA and the sliding-window cache are not built. MLA appears only as a byte count.
- No latency percentiles, TTFT under load, or goodput were measured. Those need a request stream, which this lab does not generate.
- Hardware counters were not used, so "memory-bound" rests on achieved bytes per second, not on profiler data.
