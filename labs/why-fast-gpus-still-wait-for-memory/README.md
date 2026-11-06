# Lab: why fast GPUs still wait for memory

A code companion for the post
[Why Fast GPUs Still Wait for Memory](https://srini-anirudh.github.io/blogs/why-fast-gpus-still-wait-for-memory/).
The post derives the roofline model and uses it to explain GEMM, fusion, attention and the optimizer step. This lab
measures each of those claims on a real H100 and puts every measurement next to the prediction from the post.

The package `rooflab` has four parts:

- `model.py` holds the post's formulas for W (FLOPs), Q (bytes) and the roofline. Tests check that they reproduce the
  post's numbers.
- `tilesim.py` counts the bytes a tiled GEMM moves, with an LRU cache standing in for L2.
- `kernels.py` holds three Triton kernels. Each one isolates one term of the model.
- `timing.py` times GPU work with CUDA events. `report.py` writes every output to `results/`, headed by the
  hardware and the command.

## Setup

From the repository root, after `pip install -e ".[dev]"`. The GPU scripts also need CUDA and Triton (tested with
Triton 3.9.0). The CPU scripts and tests need only PyTorch.

## Run everything

```bash
# Tests.
python -m pytest labs/why-fast-gpus-still-wait-for-memory -m "not slow and not gpu and not network"   # CPU
python -m pytest labs/why-fast-gpus-still-wait-for-memory -m gpu                                      # CUDA + Triton

# CPU scripts: the post's analytical numbers, and the GEMM traffic counter.
python labs/why-fast-gpus-still-wait-for-memory/scripts/01_ridge_points.py
python labs/why-fast-gpus-still-wait-for-memory/scripts/02_tile_traffic.py

# GPU scripts, in order. 03 and 04 measure the ceilings that 05 to 09 compare against.
bash labs/why-fast-gpus-still-wait-for-memory/scripts/run_gpu.sh
```

`run_gpu.sh` runs scripts 03 to 09 one after another. Each script can also run on its own. Without the JSON files
from 03 and 04, scripts 05 to 09 fall back to the spec-sheet ceilings and say so in their first line. The whole GPU
suite takes about 4 minutes on one H100.

## Which script covers which part of the post

| Post section | Script |
|---|---|
| One operation, two costs; arithmetic intensity; ridge points (Table 1) | `01_ridge_points.py` |
| Compute time, and why peak lies | `04_compute_ceilings.py` |
| Memory time; measure your own ceilings | `03_bandwidth.py` |
| Deriving the roofline; the overlap assumption (Figure 04) | `05_roofline_sweep.py` |
| GEMM from first principles; tiles set the intensity (Figures 06 and 07) | `02_tile_traffic.py` |
| Shapes matter; decode is a GEMV (Table 2, Figure 08) | `06_gemm_shapes.py` |
| Elementwise operations (Table 3) | `07_elementwise_fusion.py`, part 1 |
| Fusion; the fusion bench (Lab 3) | `07_elementwise_fusion.py`, part 2 |
| Attention; FlashAttention is an IO algorithm; KV cache and GQA | `08_attention.py` |
| Optimizers as memory workloads | `09_optimizers.py` |
| The third regime: latency (Little's law) | `03_bandwidth.py`, and the small GEMMs in `06_gemm_shapes.py` |
| The next wall (link ridges) | `01_ridge_points.py` |

Tutorial C3 (`c03-attention/`) builds the attention variants step by step. This lab does not rebuild them. It only
counts their bytes and times them.

## Results

Every number names its hardware. Raw outputs are in `results/`, under the name of the script that wrote them.

Tests: 19 passed on CPU (AMD EPYC 7R13, 8 threads, torch 2.15.0), with the GPU module skipped. 22 `gpu` tests passed
on NVIDIA H100 80GB HBM3 (torch 2.15.0+cu132, CUDA 13.2, Triton 3.9.0). There are no `network` or `slow` tests.

### NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2, Triton 3.9.0

One run of each script, on a GPU locked for this lab alone. Timings are medians of 5 repetitions.

| Measurement | Post says | Measured |
|---|---|---|
| HBM copy bandwidth, 1 to 4 GiB buffers | real streaming lands near 85% of 3.35 TB/s | 3.04 TB/s, 91% of spec |
| Copy with a 32 MiB working set (fits in the 50 MiB L2) | on-chip boundaries are faster | 3.70 TB/s, above the HBM spec |
| Bytes in flight needed to saturate HBM | about 2.0 MB | 2.1 MB in flight reaches 91% of measured beta |
| bf16 tensor-core matmul, n = 8192 | 989 TFLOP/s peak, real clocks lower | 702 TFLOP/s, 71% of spec |
| fp8 e4m3 / tf32 matmul, n = 8192 | 1979 / 495 TFLOP/s | 1253 / 383 TFLOP/s |
| fp32 CUDA cores: Triton FMA kernel / fp32 matmul without TF32 | 67 TFLOP/s | 63.6 / 51.6 TFLOP/s |
| Cliff from bf16 tensor cores to fp32 CUDA cores | 14.8x | 11.0x |
| FMA kernel, 0 to 48 FMAs per element | extra FLOPs are free when memory-bound | time flat at 0.357 ms |
| FMA kernel on the bandwidth diagonal (I = 2 to 12) | P = beta x I | 99% of the measured roof |
| FMA kernel at the ridge (I = 20) | between the overlapped roof and the serial model (50%) | 82% of the roof |
| d = 8192 bf16 projection, 1 to 128 tokens | step time flat until about B* | 47 to 49 us; bends at 256 (measured B* = 231) |
| Batch-1 GEMV | pure bandwidth quotient | weight read at 2.86 TB/s, 94% of measured beta |
| Square bf16 GEMM, n = 256 to 2048 | memory- or compute-bound by n/3 | 23 to 32 us each, set by kernel latency, not by either roof |
| Elementwise ops on 1.07 GB tensors (add, cast, SiLU, GELU) | t = bytes / beta | 0.98 to 1.02 of the model |
| Softmax / `F.rms_norm` on the same tensors | one-pass model | 2.95x / 1.51x the model: these kernels make extra passes |
| Fusion chain, all 16 boundary choices | time follows HBM passes | every row within about 1% of the traffic model |
| Fusion chain, all split to fully fused | 12 passes to 5, so 2.40x | 2.41x; fused kernel at 3.06 TB/s |
| Same chain: eager PyTorch / torch.compile / Triton fused | | 3.85 / 1.80 / 1.75 ms |
| Naive attention, 16 heads, d = 128, N = 2048 to 16384 | capped near beta x 64 = 195 TFLOP/s | 77 to 111 TFLOP/s |
| SDPA flash attention, same shapes | intensity grows with N | 328 to 352 TFLOP/s, 3.0x to 4.6x faster than naive |
| Decode attention, 1.07 GB KV cache, GQA g = 1 to 8 | time follows KV bytes, not FLOPs | 8x the FLOPs in 1.18x the time; 2.80 TB/s at g = 1 |
| Fused SGD / SGD momentum / AdamW, 268 M fp32 parameters | time proportional to bytes of state | 1.08 / 1.82 / 2.58 ms, all at 2.91 to 3.00 TB/s |
| AdamW / SGD | 28 / 12 bytes = 2.33x | 2.40x the time |
| Muon Newton-Schulz, one 4096 x 4096 bf16 matrix | compute-bound | 567 TFLOP/s (81% of measured bf16 pi), 3.64 ms vs 0.161 ms for AdamW |

How to read the table:

- Rows "of pi" above 100% in `06_gemm_shapes.txt` (up to 104%) are clock variation between runs. The bf16 ceiling
  comes from one n = 8192 run, so large GEMMs in a later run can beat it by a few percent.
- The flash attention plateau near 340 TFLOP/s is half the bf16 ceiling. It is FlashAttention-2 through PyTorch SDPA,
  not the Hopper-specific FlashAttention-3. Why it plateaus there was not investigated.
- The square GEMMs from n = 256 to 2048 are the post's "below both" case. Their roofline floors are under 25 us, but
  each launch costs about 23 to 32 us of fixed latency.
- The Little's law implied latency in `03_bandwidth.txt` (260 to 370 ns on the rising part) uses a lower-bound count
  of bytes in flight, one BLOCK load per program. Treat it as an order of magnitude, not a latency measurement.
- Copy rows at 1 to 4 MiB are launch-bound. A Triton launch from Python costs 18.6 us here; a `torch` copy costs 7.5 us.
- Triton 3.9 enables FP fusion by default and prints a warning about it. The FMA kernel relies on it: each
  `a * c + v` compiles to one FMA, counted as 2 FLOPs.

### AMD EPYC 7R13, 8 threads, torch 2.15.0

Scripts 01 and 02 do arithmetic, not timing, so their outputs do not depend on the machine.

| Computation | Result |
|---|---|
| Ridge points, Table 1 | match the post: H100 bf16 295, fp8 591, fp32 20 |
| Square bf16 GEMM reaches the H100 ridge at | n = 886 (post: 885) |
| Tiled intensity, 128 x 128 / 256 x 256 tiles | 64 / 128 FLOP/B |
| 70B decode at batch 1 | 41.8 ms, 24 tokens/s; B* = 295 |
| KV cache, Llama-3-70B shape, 8k context | 320 KiB per token, 2.68 GB per sequence, crossover at batch 52 |
| Tiled GEMM traffic, counted vs formula, 4096^3 | identical for 32 to 256 tiles |
| 4096^3 GEMM, 128 x 128 tiles, 132 in flight, 1 MiB LRU cache: row order / grouped order | 3.41x / 2.62x compulsory traffic |
| Same with a 50 MiB cache | 1.00x compulsory in both orders |

The post rounds the KV-cache numbers to 2.6 GB per sequence and a crossover at about 53. The exact arithmetic gives
2.68 GB and 52.

## What was not measured

- Hardware byte counters. All Q values are compulsory or modelled counts. The post's "measured traffic" accounting
  needs Nsight Compute (`dram__bytes.sum`), which this lab does not run.
- L2, shared-memory and register-file bandwidths. Only the L2 effect on copy bandwidth appears, as a side effect of
  buffer size. The hierarchical roofline (Table 4) is not measured.
- NVLink, PCIe and network ridges. They are recomputed from published rates in `01_ridge_points.py`, not measured.
- The LRU cache model in `tilesim.py` is far simpler than real L2, which is sectored and partitioned. It shows the
  direction of the effect, not real hit rates.
- Large-scale claims: the 70B decode step, the 7B Adam step and the KV crossover are arithmetic only. The "7B step"
  column in `09_optimizers.txt` scales a 268 M parameter measurement linearly.
- Whether Muon "costs little additional wall time" in a real training step. Only the Newton-Schulz kernel was timed.
- fp8 for anything except one matmul. Other GPUs in the post's tables (A100, H200, B200, MI300X, L40S, RTX 4090).
