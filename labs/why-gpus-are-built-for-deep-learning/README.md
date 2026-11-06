# Lab: why GPUs are built for deep learning

Code companion to [Why GPUs Are Built for Deep Learning](https://srini-anirudh.github.io/blogs/why-gpus-are-built-for-deep-learning/).

The post builds the GPU from one choice: spend the transistors on throughput, not latency. This lab measures that
choice on an H100, next to a CPU. Small CUDA kernels, compiled at run time with NVRTC, each isolate one mechanism.
Every measurement sits next to the post's own simplified model, written as a plain Python function in
`whygpu/models.py`. The models are tested on any CPU. The measurements need a CUDA GPU.

## Layout

```
labs/why-gpus-are-built-for-deep-learning/
├── whygpu/
│   ├── models.py    # the post's models: Little's Law, SIMD tail, divergence passes, occupancy (CC 9.0),
│   │                #   sectors, bank conflicts, counted tiled matmul, arithmetic intensity
│   ├── kernels.py   # CUDA C sources, NVRTC compile, driver queries (registers, occupancy, clock)
│   └── hw.py        # hardware label, timing (CUDA events or wall clock), result files
├── scripts/         # 00 to 09, one per mechanism, listed below
├── results/         # what the scripts wrote; each file starts with its hardware and command
└── tests/           # pytest: models, offline NVRTC compile of every kernel, scripts, GPU checks
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Only PyTorch with CUDA and numpy are needed. NVRTC ships
with the PyTorch CUDA wheels, so no `nvcc` is required.

## Run everything

```bash
L=labs/why-gpus-are-built-for-deep-learning
python -m pytest $L -m "not slow and not gpu and not network"   # CPU only, a few seconds
python -m pytest $L -m gpu                                      # needs CUDA

python $L/scripts/00_models.py                                  # the post's interactive labs as tables
for s in 01_latency_vs_throughput 02_memory_ladder 03_divergence 04_occupancy 05_coalescing \
         06_bank_conflicts 07_tiling 08_tensor_cores 09_intensity; do
  python $L/scripts/$s.py --device cuda                         # each well under a minute on an H100
done
for s in 01_latency_vs_throughput 05_coalescing 07_tiling 08_tensor_cores 09_intensity; do
  python $L/scripts/$s.py --device cpu                          # CPU baselines
done
```

## Which script covers which section of the post

| Post section | Script | Model it is checked against |
|---|---|---|
| Two answers to one question, Little's Law | `01_latency_vs_throughput.py` | `littles_law_concurrency` |
| SIMD: the CPU's answer to width (Figure 05) | `00_models.py` | `simd_loop` |
| SIMT: warps, lockstep, divergence (Figures 08, 09) | `03_divergence.py` | `divergence_passes`, `tripcount_cost` |
| Latency hiding and occupancy (Figures 10, 11) | `04_occupancy.py` | `occupancy` |
| The GPU memory hierarchy (Figure 12) | `02_memory_ladder.py` | Little's Law for bytes in flight |
| Coalescing (Figure 13) | `05_coalescing.py` | `warp_sectors` |
| Shared memory and its 32 banks (Figure 14) | `06_bank_conflicts.py` | `bank_conflict_degree`, `tile_access` |
| Tiling: manufacturing reuse (Figure 15) | `07_tiling.py` | `counted_matmul`, `matmul_traffic` |
| Tensor cores (Figure 16) | `08_tensor_cores.py` | fp16 versus fp32 accumulation |
| When does more compute stop helping? (Figure 17) | `09_intensity.py` | `op_cost`, `ridge_point` |

The next post's lab, [why-fast-gpus-still-wait-for-memory](../why-fast-gpus-still-wait-for-memory/), builds the
full roofline. `09_intensity.py` only reproduces this post's Figure 17.

## Results

Every number names its hardware. Raw output is in `results/`. Each file's first line is the hardware and its second
line is the command.

- **GPU**: NVIDIA H100 80GB HBM3 (SXM), torch 2.15.0+cu132, CUDA 13.2. One GPU, taken under an exclusive lock, so no
  other job shared it.
- **CPU**: AMD EPYC 7R13, 8 threads, torch 2.15.0+cu132. The machine was shared (load average near 90 on 192 CPUs),
  so treat CPU timings as orders of magnitude.

Spec figures (989 TFLOP/s dense bf16, 67 TFLOP/s fp32, 3.35 TB/s) are NVIDIA's published H100 SXM5 numbers. They
are not measurements.

### Latency, throughput and Little's Law (`01`, `02`)

| Measurement | Hardware | Result |
|---|---|---|
| One dependent fp32 FMA chain, one thread | H100 | 4.44 cycles per FMA (loop unrolled 16x, so this includes a little loop overhead) |
| FMAs in flight that Little's Law asks for: 4.44 cycles x 16,896 lanes | H100 | 74,979 |
| FMA throughput at 0.23x / 0.9x / 1.8x / 7.2x that many threads | H100 | 20% / 70% / 83% / 100% of the best rate |
| Best fp32 FMA rate (540,672 threads) | H100 | 51.9 TFLOP/s, against 67 spec |
| 2,000 dependent one-element ops launched from Python | H100 / CPU | 6.65 / 3.55 us per op |
| `z = x + y`, 2^14 elements | H100 / CPU | 3.4 / 4.0 G elements/s |
| `z = x + y`, 2^28 elements | H100 / CPU | 257 / 4.6 G elements/s (3,089 / 56 GB/s) |
| Pointer chase, L1 (16 KB, plain load) | H100 | 39 cycles, 20 ns |
| Pointer chase, L2 (16 KB to 32 MB, `__ldcg`) | H100 | 284 cycles, 143 ns |
| Pointer chase, HBM (256 MB / 2 GB) | H100 | 693 / 701 cycles, 350 / 353 ns |
| Bytes in flight Little's Law asks for: 350 ns x 3.35 TB/s | H100 | 1.17 MB, 8.7 KB per SM |

The ranking inverts exactly as in Figure 01. For a chain of dependent ops, and for arrays up to about 2^14 elements,
the CPU wins. Past that, the GPU needs more work before its time per call moves at all: below 2^20 elements a GPU
add costs about 4.8 us however large it is. Throughput reaches 70% when the threads in flight match Little's Law and
saturates at a few times more. The measured HBM latency, 350 ns, matches the post's ~350 ns. L2 at 284 cycles is
slower than the post's figure suggests. An L2 hit is still half the latency of HBM.

### Divergence (`03`)

H100, 2,162,688 threads, each running one path body of 2,048 dependent FMAs.

| Paths | Mapping | Model slowdown | Measured slowdown |
|---|---|---|---|
| 2 / 4 / 8 | path = (tid / 32) % P, warp-aligned | 1 / 1 / 1 | 1.00 / 1.01 / 1.00 |
| 2 / 4 / 8 | path = tid % P | 2 / 4 / 8 | 1.91 / 3.71 / 7.36 |
| 2 / 4 / 8 | random path per thread | 2 / 4 / 7.89 | 1.90 / 3.71 / 7.33 |
| trip count uniform in [16, 4096], same per block vs shuffled | | 1.94 | 1.74 |

The same code, with the same work per thread, runs 7.4 times slower when eight paths are interleaved inside each
warp. The measured slowdown stays a little under the model, because a fixed cost (launch, the switch, the store)
is not serialised.

### Occupancy and latency hiding (`04`)

**Part A: the calculator against the driver.** The test kernel was compiled at several register caps, which gave
32, 40, 64, 72 and 102 registers per thread. For each of 138 configurations, the lab asks the driver's own
calculator (`cuOccupancyMaxActiveBlocksPerMultiprocessor`) how many blocks fit.

| Model | Matches the H100 driver |
|---|---|
| The post's Figure 11 calculator | 119 of 138 |
| The same calculator plus two driver rules | 138 of 138 |

The two rules the post's calculator leaves out:

- The driver reserves 1 KB of shared memory per block. At 57 KB per block, 228 / 57 gives 4 blocks, but the driver
  fits 3.
- Registers are allocated per processing block, and each processing block holds a quarter of the 65,536 registers in
  whole warps. At 102 registers, a warp needs 3,328 registers. That fits 19 warps in the whole SM, but only 4 per
  quarter, 16 in all. So 64-thread blocks fit 8 per SM, not 9.

**Part B: Volkov's point.** Streaming 1 GiB on the H100, 256 threads per block. Shared memory was requested only to
cap the number of blocks per SM.

| Loads in flight per thread (ILP) | 12.5% occupancy | 25% | 50% | 100% |
|---|---|---|---|---|
| 1 | 1,261 GB/s | 2,190 | 2,901 | 3,030 |
| 2 | 2,003 | 2,812 | 3,044 | 3,049 |
| 4 | 2,725 | 3,021 | 3,071 | 3,075 |
| 8 | 2,743 | 3,028 | 3,065 | 3,067 |

With 4 independent 16-byte loads per thread, 25% occupancy already reaches 3.02 TB/s, 90% of spec. One load per
thread at 25% occupancy gets 2.19 TB/s. Bytes in flight decide the result, and occupancy is only one way to get
them. At 12.5% occupancy with ILP 4, an SM keeps 256 x 4 x 16 = 16 KB in flight. That is about twice the 8.7 KB
that Little's Law asks for at 350 ns (`02`), and it reaches 81% of spec.

### Coalescing (`05`)

Thread i reads element i x stride: 67,108,864 useful fp32 loads (256 MiB). "Implied" is the useful bandwidth divided
by the model's efficiency, which estimates the bytes the memory system really moved.

| Pattern | Sectors per warp (model) | Model efficiency | H100 useful GB/s | H100 implied GB/s |
|---|---|---|---|---|
| stride 1 | 4 | 100% | 2,093 | 2,093 |
| stride 2 | 8 | 50% | 1,398 | 2,796 |
| stride 4 | 16 | 25% | 761 | 3,042 |
| stride 8 | 32 | 12.5% | 391 | 3,126 |
| stride 16 | 32 | 12.5% | 200 | 1,598 |
| stride 32 | 32 | 12.5% | 175 | 1,398 |
| stride 1, base shifted 4 bytes | 5 | 80% | 2,070 | 2,588 |
| random gather | 32 | 12.5% | 121 | 965 |

| Pattern, numpy on one CPU thread | 64-byte-line efficiency | CPU useful GB/s |
|---|---|---|
| stride 1 / 2 / 4 / 8 | 100% / 50% / 25% / 12.5% | 17.4 / 11.3 / 3.6 / 2.3 |
| stride 16 / 32 | 6.25% / 6.25% | 0.89 / 0.45 |

From stride 1 to stride 8, useful bandwidth halves with each doubling, as the sector model says. The implied
traffic stays between 2.1 and 3.1 TB/s, so the memory system is busy the whole time, mostly moving bytes nobody uses.
Strides 16 and 32 fall another 2x below the model. The model stops at 32 sectors, but DRAM and the TLB work in coarser
units, which it does not count. The shifted base costs nothing measurable: the extra sector is shared with the next
warp through L2, as the post's caption for Figure 13 warns. This simple kernel issues 4-byte loads, and at stride 1
it reaches 62% of spec. The 16-byte loads in `04` reach 92%.

### Shared-memory bank conflicts (`06`)

H100, one block of 32 warps per SM, each lane reading one fixed shared word 8,000 times.

| Access on a 32 x 32 float tile | Model conflict degree | Measured cycles per warp access |
|---|---|---|
| a row | 1 | 1.00 |
| a column | 32 | 32.0 |
| a column, rows padded to 33 words | 1 | 1.00 |
| stride 2 / 4 / 8 / 16 | 2 / 4 / 8 / 16 | 2.00 / 4.00 / 8.00 / 16.0 |
| stride 3 | 1 | 1.00 |
| every lane the same word (broadcast) | 1 | 1.00 |

The model is exact here. A column read costs 32 cycles. One word of padding makes it cost 1. One warp access per
cycle per SM is 128 bytes per cycle, the shared-memory bandwidth in the post's Figure 12.

### Tiling (`07`)

The counted matmul (pure Python, n = 32) loads exactly 2n^3 / T elements for every T, and gives the same C as the
naive loop (largest difference 9e-15). On hardware:

| Kernel, n = 4096, fp32 | Hardware | ms | TFLOP/s |
|---|---|---|---|
| naive CUDA, one thread per C element | H100 | 26.0 | 5.3 |
| shared-memory tile T = 8 / 16 / 32 | H100 | 27.5 / 17.2 / 15.0 | 5.0 / 8.0 / 9.2 |
| cuBLAS `torch.mm`, TF32 off | H100 | 2.66 | 51.7 |
| `torch.mm`, n = 2048 | CPU | 33.2 | 0.52 |

Tiling through shared memory makes the kernel 1.7 times faster at T = 32. That is far less than the 32-fold drop in
loads, because the naive kernel already gets reuse from L1 and L2, which this count leaves out. cuBLAS is another
5.6 times faster. It also tiles in registers, uses wider loads and pipelines its loads. These kernels show the
mechanism. They are not tuned.

### Tensor cores and accumulation (`08`)

| Measurement | Hardware | Result |
|---|---|---|
| Length-4,096 dot product of fp16 values, summed in fp16 | CPU (numpy, exact) | relative error 8.3e-2 |
| The same, summed in fp32 | CPU (numpy, exact) | relative error 4.5e-7 |
| fp16 matmul with K = 4,096, against fp64 | H100 | largest relative error 5.0e-4, the size of one fp16 rounding of the output |
| Dense matmul n = 8192: fp32 / TF32 / bf16 / fp16 | H100 | 51.9 / 371.7 / 777.7 / 665.0 TFLOP/s |
| The same as a share of the dense spec peak | H100 | 77% / 75% / 79% / 67% |
| bf16 matmul n = 4096 / 4090 / 4088 | H100 | 670 / 201 / 661 TFLOP/s |
| bf16 matmul n = 8192 / 8190 / 8184 | H100 | 681 / 274 / 682 TFLOP/s |
| `torch.mm` n = 1024, fp32 / bf16 | CPU | 643 / 1.0 GFLOP/s |

Summing in fp16 loses 8% of a 4,096-term dot product. Summing in fp32 loses less than one part in a million. The GPU
matmul's error is the size of a single fp16 rounding, which shows that its sum was not kept in fp16. Shape matters
as much as the post says. n = 4090 runs 3.3 times slower than 4096, while 4088, a multiple of 8, runs at full speed.
The bf16 and fp16 rates differ by 15% in this single run. Treat that gap as unexplained. This CPU has no bf16 matrix
hardware, so its bf16 matmul is 600 times slower than its fp32 one.

### Arithmetic intensity (`09`)

H100, fp16 tensors of 8192 x 8192. FLOP counts are the post's hand counts (whygpu/models.py). Bytes are the minimum.

| Op | FLOP/byte | TFLOP/s | Share of 989 TFLOP/s | GB/s | Share of 3.35 TB/s |
|---|---|---|---|---|---|
| copy | 0 | 0 | 0% | 2,916 | 87% |
| elementwise add | 0.17 | 0.50 | 0.1% | 3,010 | 90% |
| GELU (tanh) | 2.5 | 7.3 | 0.7% | 2,907 | 87% |
| LayerNorm | 2.0 | 3.7 | 0.4% | 1,867 | 56% |
| softmax | 1.25 | 1.2 | 0.1% | 995 | 30% |
| matmul 8192^3 | 2,731 | 753 | 76% | 276 | 8% |

The ridge point is 295 FLOP per byte. Every op below it reaches at most 0.7% of the tensor-core peak, even when it
runs at 90% of HBM bandwidth. The matmul is above the ridge, and it is the only op that gets near the peak. On the
CPU (fp32, 2048 x 8192), the same ops run at 5 to 8 GFLOP/s and the matmul at 684 GFLOP/s (`09_intensity_cpu.txt`).

## What was not measured

- **Die area (Figure 02)** is a spec and schematic figure. Nothing here measures area.
- **CPU pipeline hazards, out-of-order execution and FMA latency (Figure 04).** Python cannot time single CPU
  instructions. The CPU side of this lab is limited to whole PyTorch and numpy operations.
- **CPU memory latency** (the post's ~80 ns) is not measured. Only the GPU ladder is.
- **Actual DRAM traffic.** No profiler (Nsight Compute) was used. Traffic numbers are model counts, and
  bandwidth is inferred from time.
- **Tensor-core register-file amortisation (Figure 16)** is not isolated. Only end-to-end matmul rates are shown.
- **The fp16 accumulation demo** sums sequentially. Real fp16 kernels that accumulate in fp16 use different orders,
  so their error differs.
- Each GPU timing is one run of the script, as a median of 5 trials. Run-to-run spread was not studied, except where
  noted.


