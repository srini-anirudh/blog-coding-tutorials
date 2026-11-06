# Lab: How to Find What Is Slowing Your Model

Runnable companion to the post
[How to Find What Is Slowing Your Model](https://srini-anirudh.github.io/blogs/how-to-find-what-is-slowing-your-model/).

The post argues that every optimisation is a hypothesis about the bottleneck, and that a few cheap measurements
decide which hypothesis is true. This lab makes those measurements in small Python scripts and runs them on an
H100. You can see each claim happen:

- The GPU busy fraction, read from a PyTorch profiler trace, separates a launch-bound step from a GEMM-bound one.
- The NVML utilisation counter (the number `nvidia-smi` prints) reads about the same for a tiny add and a large
  matmul.
- Three numbers (FLOPs, bytes, time) give a roofline verdict. Fusion moves a point; a time budget says if it matters.
- A bounded prefetch queue runs dry when the producer is slower than the consumer. A real DataLoader agrees with
  the model.
- A slow all-reduce is a fast rank waiting. A barrier splits arrival skew from transfer.
- An out-of-memory error can happen while a gigabyte sits cached and idle. The OOM message says which problem it is.
- Decode time is flat in batch size until the KV cache reads catch up with the weight reads.
- The whack-a-mole case study: fix the measured bottleneck, re-measure, repeat, and stop.

The package is `whyslow/`. It has no dependencies beyond PyTorch.

## Layout

```
how-to-find-what-is-slowing-your-model/
├── whyslow/
│   ├── timeline.py    # lanes as intervals: busy fraction, gaps, exposed time, Amdahl; chrome-trace reader
│   ├── profile.py     # torch.profiler with a schedule, returning the triage numbers
│   ├── roofline.py    # verdict from FLOPs, bytes and time; H100 SXM vendor peaks; time budget
│   ├── queue_sim.py   # producer, bounded queue and consumer, as in torch.utils.data
│   ├── memory.py      # allocated / reserved / peak, and an OOM-message parser
│   ├── toy.py         # residual MLP and a training step
│   └── bench.py       # CUDA-event timing, NVML utilisation via ctypes, results files with a hardware line
├── scripts/           # one script per idea, listed below; run_gpu.sh runs every GPU script
├── results/           # real outputs; each file starts with its hardware and command
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Scripts 01, 02, 03, 06, 07 and 08 need a CUDA GPU.
Without one they print that nothing was measured and exit. Scripts 04 and 05 run on CPU.

## Run everything

```bash
cd labs/how-to-find-what-is-slowing-your-model

# Tests. Offline ones take a few seconds.
python -m pytest . -m "not slow and not gpu and not network"
python -m pytest . -m slow      # runs scripts 04 (quick) and 05 end to end, about 30 s; rewrites results/05_*
python -m pytest . -m gpu       # needs CUDA

# CPU scripts.
python scripts/04_input_pipeline.py      # about 40 s, mostly sleeping
python scripts/05_distributed_skew.py    # 4 gloo processes, about 20 s

# GPU scripts, one after another. About 5 minutes on one H100.
bash scripts/run_gpu.sh
```

## Which script covers which section of the post

| Post section | Script |
|---|---|
| The timeline is the primitive; triage step 1 (sum of kernel time / wall time) | `01_busy_fraction.py` |
| Trap: the profiler changes the thing it measures | `01_busy_fraction.py` (profiled vs unprofiled step) |
| Eleven ways to be slow: kernel latency, CPU launch overhead | `01_busy_fraction.py`, `08_whack_a_mole.py` |
| What GPU utilisation actually counts (Figure 02) | `02_utilisation_lies.py` |
| Three numbers into a verdict (Figure 04); Trap: a roofline point without a time budget | `03_roofline.py` |
| The input pipeline (Figure 05); Nuance: time `next(iter)` | `04_input_pipeline.py` |
| Distributed training (Figure 06): barrier split, per-rank spread, halve the message | `05_distributed_skew.py` |
| Memory diagnosis (Figure 07); Aside: read the OOM message | `06_memory.py` |
| Inference diagnosis: decode is bandwidth bound; batching is nearly free until it isn't | `07_decode_batching.py` |
| Overlap, and what a step really costs (Lab 02 arithmetic) | `whyslow/timeline.py`: `exposed`, `step_time` (tests) |
| Case study: whack-a-mole (Figure 09) | `08_whack_a_mole.py` |
| Ten questions: Amdahl's 30% -> 1.43x | `whyslow/timeline.py`: `max_speedup` (tests) |

## Results

Every number names its hardware. Raw output is in `results/`, one `.txt` and one `.json` per script, each starting
with its hardware line and command. GPU runs used one NVIDIA H100 80GB HBM3 (SXM), torch 2.15.0, CUDA 13.2, with a
lock that kept other jobs off that GPU. The host CPU was shared, so CPU-bound timings (launch overhead) are noisier
than GPU-bound ones. Peaks quoted as "vendor" are the H100 SXM datasheet numbers: 989 TFLOP/s dense bf16 and
3.35 TB/s HBM3.

Tests on AMD EPYC 7R13, 8 threads, torch 2.15.0: 19 offline passed; 2 `slow` passed; 3 `gpu` passed on the H100.
There are no `network` tests. See `results/tests.txt`.

### 01: the busy fraction separates two kinds of slow (H100)

Both are training steps of the same residual MLP, depth 12, AdamW, with a `loss.item()` each step.

| | small: width 256, batch 64, fp32 | large: width 4096, batch 8192, bf16 |
|---|---|---|
| Step time, unprofiled | 11.29 ms | 177.66 ms |
| Step time, under `torch.profiler` | 20.94 ms (+85%) | 178.60 ms (+1%) |
| GPU busy per step (kernel time from the trace) | 1.71 ms | 176.92 ms |
| GPU busy / unprofiled step | 0.152 | 0.996 |
| Kernels per step; mean duration | 818; 2.1 us | 919; 192.4 us |
| Model TFLOP/s (6 x weights x batch) | 0.21 | 445.6 |

The small step is CPU launch overhead: 818 kernels of about 2 us each, and the GPU idle 85% of the time. No
kernel optimisation can buy it more than 1 / 0.152 = 6.6x. The large step is dense GEMM, and gaps are not its
problem. The profiler nearly doubled the small step and left the large one alone. That is the post's first Trap:
tracing adds CPU work to the lane that is already the problem.

### 02: NVML utilisation reads 100% for all three kernels (H100)

| Kernel | NVML `utilization.gpu` | Achieved | % of vendor peak |
|---|---|---|---|
| A: `add_` on 4096 floats, 200 per CUDA graph | 100% | 0.004 TFLOP/s, 32 GB/s | 0.00% FLOP, 1.0% HBM |
| B: elementwise multiply on 1 GiB fp32 | 100% | 3030 GB/s | 90.5% HBM |
| C: 8192^3 bf16 matmul | 100% | 680 TFLOP/s | 68.8% FLOP |

The counter was sampled every 50 ms through `libnvidia-ml` while each kernel ran for 4 s. It cannot tell the three
apart. Bytes and FLOPs are analytic.

### 03: three numbers into a verdict (H100)

| Case | Time | Intensity (FLOP/B) | Achieved | % of roof | Verdict |
|---|---|---|---|---|---|
| 64^3 bf16 matmul | 39.3 us | 21 | 0.01 TFLOP/s | 0.0% | below both roofs |
| Elementwise chain, 4 kernels, 64 Mi bf16 | 506 us | 0.50 | 2917 GB/s | 87% | bandwidth bound |
| Same chain, `torch.compile` (1 kernel) | 211 us | 1.38 | 2548 GB/s | 76% | bandwidth bound |
| 16 x 8192 @ 8192 x 8192 (decode-like) | 71.9 us | 16 | 1873 GB/s | 56% | bandwidth bound |
| 256 x 8192 @ 8192 x 8192 | 77.0 us | 241 | 446 TFLOP/s | 55% | bandwidth bound |
| 8192^3 bf16 matmul | 1433 us | 2731 | 768 TFLOP/s | 78% | compute bound |

The ridge is 295 FLOP/byte. Fusion moved the chain to the right (2.75x fewer bytes) and cut its time 2.4x, with the
same FLOPs. Measured ceilings on this card: 2.93 TB/s for a 512 MiB copy (read + write) and 752 TFLOP/s for the
8192^3 matmul. FLOPs and bytes are analytic, and the GELU is counted as 8 FLOPs per element by convention. The
time-budget arithmetic matches the post: tripling a 4% part saves 2.7% of the step, halving an 18% part saves 9%.

### 04: a queue that runs dry (AMD EPYC 7R13, 8 threads)

GPU step 80 ms; `prefetch_factor` 2; each worker takes 384 ms per batch.

| Config | Simulation: wait per step | Simulation: time lost | Real DataLoader: wait in `next()` | Real: share of time waiting |
|---|---|---|---|---|
| 4 workers (one batch per 96 ms) | 16.0 ms | 16.7% | 16.3 ms | 16.7% |
| 8 workers (one batch per 48 ms) | 0.0 ms | 0.0% | 3.0 ms | 3.5% |

With a full queue at the start, configuration A runs dry after 28 steps and stays dry. In steady state its depth
histogram has 25% of its mass at zero. The 16 ms arrives as about 64 ms every fourth step, because the four workers
finish together. With 8 workers the ready count sits at 12 of 16 slots; the rest are batches in progress. The
real loader agrees with the model to within 0.3 ms for A. The 3.0 ms for 8 workers is worker start-up and scheduling
noise on a shared host. A local run of the same script measured 1.1 ms.

### 05: a slow all-reduce is a fast rank waiting (AMD EPYC 7R13, 8 threads, gloo, 4 processes)

Each rank sleeps 40 ms (its compute), then all-reduces 16 MiB. Medians of 12 steps.

| Scenario | Rank | All-reduce alone | Barrier | All-reduce after the barrier |
|---|---|---|---|---|
| Balanced | 0 to 3 | 21.5 to 21.8 ms | 0.6 to 0.8 ms | 20.2 to 20.6 ms |
| Rank 2 computes 18 ms longer | 0, 1, 3 | 36.3 to 36.4 ms | 18.6 to 18.7 ms | 17.7 ms |
| Rank 2 computes 18 ms longer | 2 | 18.3 ms | 0.6 ms | 17.8 ms |

The fast ranks see a slow all-reduce. The barrier shows that 18.6 of their 36.4 ms is waiting for rank 2. The
transfer itself did not change. The per-rank compute spread (18.0 ms) names the cause. Halving the message from
64 MiB to 32 MiB to 16 MiB took 62.6, 47.3 and 18.4 ms: bandwidth bound, if noisy. Halving 4 KiB to 2 KiB took
1.03 and 0.98 ms: latency bound. This is CPU shared memory, not NCCL or a network.

### 06: an OOM with 1.25 GiB cached and idle (H100)

Part 1: after a 2 GiB temporary is freed, allocated drops to 1.00 GiB and reserved stays at 3.00 GiB. Peak stays at
3.00 GiB. `empty_cache()` returns reserved to 1.00 GiB.

Part 2 caps the process at 2 GiB with `set_per_process_memory_fraction`, allocates 400 tensors of 4 MiB, keeps 80,
then asks for 1 GiB.

| Allocation order and allocator | Allocated | Reserved | Cached but idle | 1 GiB request |
|---|---|---|---|---|
| Interleaved (one kept in every five), default | 0.31 GiB | 1.56 GiB | 1.25 GiB | OOM |
| Interleaved, `expandable_segments:True` | 0.31 GiB | 1.56 GiB | 1.25 GiB | OOM |
| The 80 long-lived tensors allocated first, default | 0.31 GiB | 1.56 GiB | 1.25 GiB | succeeds |

The device had 76.6 GiB free. The OOM message carried every number needed: requested 1 GiB, 320 MiB allocated by
PyTorch, 1.25 GiB reserved but unallocated, 2 GiB allowed. `whyslow.memory.parse_oom` reads them, and the verdict
is fragmentation. The same bytes in a stable order fit, because the temporaries then fill segments of their own,
those segments empty completely, and the allocator returns them before it retries. On this torch build
`expandable_segments` did not rescue the interleaved case. Its message says the setting was enabled.

### 07: decode is flat in batch size until the KV cache catches up (H100)

Synthetic decoder: 8 layers, width 4096, 32 heads, bf16, 3.00 GiB of weights (0.96 ms to read at vendor bandwidth).
One decode step for b sequences over a fixed KV cache of L tokens. The bytes model is (weights + KV) / 3.35 TB/s.

| Context | Batch | Step | Tokens/s | KV / weights | Bytes model | Achieved |
|---|---|---|---|---|---|---|
| 512 | 1 | 1.93 ms | 518 | 0.02 | 0.98 ms | 1702 GB/s |
| 512 | 16 | 1.95 ms | 8215 | 0.33 | 1.28 ms | 2205 GB/s |
| 512 | 32 | 2.08 ms | 15424 | 0.67 | 1.60 ms | 2588 GB/s |
| 512 | 128 | 4.25 ms | 30137 | 2.67 | 3.53 ms | 2781 GB/s |
| 4096 | 1 | 2.03 ms | 494 | 0.17 | 1.12 ms | 1856 GB/s |
| 4096 | 4 | 2.01 ms | 1989 | 0.67 | 1.60 ms | 2669 GB/s |
| 4096 | 16 | 4.10 ms | 3906 | 2.67 | 3.53 ms | 2883 GB/s |
| 4096 | 64 | 12.23 ms | 5233 | 10.67 | 11.22 ms | 3073 GB/s |

At context 512, going from 1 to 32 sequences cost 7% more time and gave 30x the tokens. At context 4096 the step
time doubled by batch 16, once the KV reads were larger than the weights. At context 4096 and batch 64 the step is
within 9% of the bytes model. At batch 1 the step is 2x the bytes model: dozens of small kernels per step, so
latency, not bandwidth.
A 4096-token prefill through the same layers took 22.4 ms at 639 TFLOP/s: compute bound. All rows are in
`results/07_decode_batching.txt`.

### 08: whack-a-mole (H100)

Residual MLP, width 1024, depth 8, batch 4096; 1650 GFLOP per step. Workers spend 120 ms per batch. MFU uses the
dense bf16 vendor peak for every stage, as the post does. The diagnosis column comes from a fixed rule in the
script: wait in `next()` above 20% of the step means input pipeline; busy fraction below 0.7 means launch gaps;
dense fp32 work means idle tensor cores; dense bf16 work means stop.

| Stage | Step | Under profiler | Wait in `next()` | Busy fraction | Kernels | MFU | Speedup | Diagnosis |
|---|---|---|---|---|---|---|---|---|
| 0 baseline: 2 workers, fp32, per-parameter AdamW | 61.01 ms | 60.90 ms | 21.57 ms | 0.61 | 530 | 2.7% | 1.00x | input pipeline |
| 1 + 32 workers, pinned memory | 37.75 ms | 37.99 ms | 0.07 ms | 0.98 | 530 | 4.4% | 1.62x | dense fp32 work |
| 2 + bf16 autocast | 13.44 ms | 19.62 ms | 0.07 ms | 0.49 | 626 | 12.4% | 4.54x | launch gaps |
| 3 + `torch.compile`, fused AdamW, loss read every 50 steps | 6.31 ms | 12.35 ms | 0.05 ms | 0.80 | 204 | 26.4% | 9.66x | stop |
| 4 + CUDA graphs (`mode="reduce-overhead"`) | 4.86 ms | 5.17 ms | 0.04 ms | 1.00 | 217 | 34.3% | 12.55x | stop |

Each fix promoted a different resource to the top, as in Figure 09. Fixing the loader made the step GPU bound. bf16
made it launch bound. Compiling made it mostly dense again. The rule said stop after stage 3. Stage 4 shows what
the remaining 20% of gaps were worth: 1.30x from CUDA graphs, at no correctness risk. Under the profiler, stages 2
and 3 looked about twice as slow, so the busy fraction divides trace kernel time by the unprofiled step. In stage 4
that ratio is slightly above 1 (4.95 / 4.86 ms) because the two numbers come from separate runs; the script caps it
at 1.00. The post's run went from 380 ms to 166 ms on a different workload. These numbers are not that run.

## What was not measured

- Nsight Systems and Nsight Compute. They are not installed on the machine used here. The lab uses the PyTorch
  profiler (scale 1 of the zoom ladder) and analytic FLOP and byte counts. Hardware counters such as DRAM
  throughput, tensor-pipe activity and warp stall reasons were not read.
- The post's large-scale figures (Figure 01's 120 ms VLM step, Figure 09's 380 ms to 166 ms run, the 64-GPU
  trap) describe workloads this lab does not run. The scripts show each mechanism at small scale. Their numbers
  are not the post's numbers.
- The distributed experiment runs four CPU processes over gloo on one machine. It shows the barrier split and the
  latency/bandwidth test. It does not measure NCCL, NVLink or a network. `--backend nccl --world 2` exists but was
  not run.
- The input-pipeline and distributed scripts use `time.sleep` for decode and compute, so their numbers do not
  depend on CPU load. Real decode cost and its variance were not measured.
- Serving schedulers (TTFT and TPOT under continuous batching, chunked prefill) were not built. Script 07 measures
  only the decode step itself.
