# Lab: The Parallelism Playbook

This lab builds tensor, sequence, pipeline and context parallelism from scratch in small PyTorch files. Each one
runs on several real processes and is checked against a single-device run of the same computation. A ledger counts
every collective a layer issues and the bytes each rank sends, so you can see the communication the post derives.

Theory post: https://srini-anirudh.github.io/blogs/the-parallelism-playbook/

Data parallelism, ZeRO and FSDP are built in [C5: pretraining](../../c05-pretraining/) (`pretrain/distributed.py`,
`pretrain/zero.py`, `pretrain/fsdp.py`). This lab does not rebuild them. It covers the axes the post adds after
FSDP: tensor, sequence, pipeline and context parallelism, the collectives under them, and the device mesh.

## What you can see happen

- A ring AllReduce built from send/recv gives the same sum as `dist.all_reduce`. ReduceScatter followed by
  AllGather gives the same result too. Each rank sends 2(N-1)/N of the buffer.
- A Megatron block split column-then-row matches one device to about 1e-6. It issues exactly 2 AllReduces forward
  and 2 backward per layer. Each one carries the activation b·s·h, so it grows with the batch. The parameter count
  does not change it.
- Splitting row-first instead needs two collectives in the MLP forward pass, because GELU of partial sums is not
  GELU of the sum.
- Sequence parallelism turns each AllReduce into a ReduceScatter and an AllGather. The layer sends the same bytes.
  The replicated layernorm inputs shrink by T. The lab also shows a detail the post leaves out: to keep the
  saving in the matmul's saved input, Megatron gathers it a second time in backward.
- A clock simulator runs GPipe, 1F1B, interleaved 1F1B and a zero-bubble-style schedule. GPipe and 1F1B match
  (P-1)/(m+P-1) exactly. 1F1B holds at most P activation stashes instead of m.
- A real pipeline runs one stage per process over send/recv. Its gradients match one device.
- Ring attention with an online softmax is exact. Causal masking leaves contiguous shards uneven, and the zigzag
  layout evens them. Ulysses transposes the layout with four All-to-Alls and runs plain attention.
- The device mesh as arithmetic: with TP varying fastest, every TP group of 8 or fewer stays inside one node.

## Layout

```
labs/the-parallelism-playbook/
├── ppbook/
│   ├── dist_utils.py       # spawn N ranks (gloo on CPU, NCCL on GPU), results files
│   ├── comm.py             # collectives with a ledger; Megatron's f, f_bar, g, g_bar; All-to-All
│   ├── ring.py             # ring ReduceScatter, AllGather and AllReduce from send/recv
│   ├── model.py            # the single-device reference block and a small stack
│   ├── tensor_parallel.py  # column/row-parallel block, sequence parallelism, the regather variant
│   ├── pipeline.py         # GPipe, 1F1B, interleaved schedules; clock simulator; send/recv executor
│   ├── context_parallel.py # ring attention (contiguous and zigzag), Ulysses
│   ├── mesh.py             # rank -> (tp, cp, dp, pp), groups, node crossings
│   └── workers.py          # per-rank checks and benchmarks
├── scripts/                # 01 to 08, listed below
├── results/                # outputs written by the scripts, each with its hardware line
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Everything except `08_bench_tp.py` and the `--bench`
modes runs on CPU. The scripts start their own processes with `torch.multiprocessing`, so no `torchrun` is needed.
Each rank uses one thread.

## Run everything

```bash
# Tests: CPU, gloo, a few seconds per test (22 tests, about 2 minutes).
python -m pytest labs/the-parallelism-playbook -m "not slow and not gpu and not network"
python -m pytest labs/the-parallelism-playbook -m gpu          # needs two CUDA GPUs

# CPU demos (about 5 minutes in total).
python labs/the-parallelism-playbook/scripts/01_collectives.py
python labs/the-parallelism-playbook/scripts/02_tensor_parallel.py
python labs/the-parallelism-playbook/scripts/03_sequence_parallel.py
python labs/the-parallelism-playbook/scripts/04_pipeline_schedules.py
python labs/the-parallelism-playbook/scripts/05_pipeline_run.py
python labs/the-parallelism-playbook/scripts/06_context_parallel.py
python labs/the-parallelism-playbook/scripts/07_mesh.py
python labs/the-parallelism-playbook/scripts/01_collectives.py --bench --world 2 --max-mb 16 --out 01_collectives_bench_cpu.txt

# Two GPUs (about 15 minutes in total).
python labs/the-parallelism-playbook/scripts/01_collectives.py --bench --world 2 --max-mb 256 --out 01_collectives_bench_gpu.txt
python labs/the-parallelism-playbook/scripts/08_bench_tp.py --world 2 --out 08_bench_tp_gpu.txt
python labs/the-parallelism-playbook/scripts/05_pipeline_run.py --bench --stages 2 --out 05_pipeline_bench_gpu.txt
python labs/the-parallelism-playbook/scripts/06_context_parallel.py --bench --world 2
```

## Which script covers which section of the post

| Post section | Script | Code |
|---|---|---|
| The collectives, briefly (AllReduce = ReduceScatter + AllGather, the α–β model) | `01_collectives.py` | `ring.py`, `comm.py` |
| Tensor parallelism (column, row, f and f̄, message size, why column first) | `02_tensor_parallel.py`, `08_bench_tp.py` | `tensor_parallel.py` |
| Tensor parallelism, "measure the GEMM" | `08_bench_tp.py` part 2 | |
| Sequence parallelism | `03_sequence_parallel.py`, `08_bench_tp.py` | `tensor_parallel.py`, `comm.g`, `comm.g_bar` |
| Pipeline parallelism, the bubble, 1F1B, interleaving, zero bubble | `04_pipeline_schedules.py` | `pipeline.simulate` |
| Pipeline parallelism, a real run | `05_pipeline_run.py` | `pipeline.run_stage` |
| Context parallelism: ring attention, causal balance, Ulysses | `06_context_parallel.py` | `context_parallel.py` |
| Combining parallelisms, topology-aware placement | `07_mesh.py` | `mesh.py` |
| Data parallelism, DDP, ZeRO, FSDP | not here: see C5 | `c05-pretraining/pretrain/` |

## Results

Every number names its hardware. Raw output is in `results/`, under the name of the script that wrote it.

Tests on AMD EPYC 7R13, gloo, torch 2.15.0: 22 passed, 3 `gpu` tests deselected, 133 s. The 3 `gpu` tests passed
on 2 x NVIDIA H100 80GB HBM3 (NCCL, CUDA 13.2). There are no `network` or `slow` tests. See `results/tests.txt`.

### Correctness and communication counts: AMD EPYC 7R13, gloo, one thread per rank, torch 2.15.0

These numbers do not depend on speed. Errors are max absolute differences in fp32 against one device.

| Measurement | Result |
|---|---|
| Ring AllReduce from send/recv vs `dist.all_reduce`, 2 / 3 / 4 / 8 ranks | max error 0 / 4.8e-7 / 9.5e-7 / 1.4e-6 |
| ReduceScatter then AllGather vs AllReduce | identical (error 0) at every world size |
| Bytes each rank sends in a ring AllReduce, as a fraction of the buffer, N = 2 / 3 / 4 / 8 | 1.000 / 1.333 / 1.500 / 1.750, which is 2(N-1)/N |
| TP block vs one device, T = 2 / 4 / 8: output, input-gradient and weight-gradient error | at most 4.8e-7, 7.2e-7 and 2.4e-6 |
| Collectives per TP layer (forward + backward), any T | 2 AllReduce forward + 2 backward, nothing else |
| Bytes per TP AllReduce, batch x seq = 2x16 / 8x16 / 8x64 (h = 64) | 8 / 32 / 128 KiB, always b·s·h·4; the block's 195 KiB of parameters never changes it |
| GELU of partial sums, summed, vs GELU of the full sum | max error 0.75: row-first must reduce before GELU |
| MLP forward collectives: column-then-row vs row-then-column (T = 2) | 1 AllReduce of 8 KiB vs 1 AllReduce of 32 KiB + 1 AllGather |
| TP vs TP+SP, T = 2, h = 256: bytes each rank sends on the layer path | 256 KiB for both |
| Activation bytes saved for backward, one rank, T = 2 / 4: TP, TP+SP, TP+SP regather (share of one device) | 53.1 / 51.6 / 50.1% and 29.7 / 27.4 / 25.1% |
| Extra communication of TP+SP regather | 2 extra AllGathers in backward per layer (+64 KiB at T = 2), plus a 6 KiB layernorm/bias gradient AllReduce for any SP |
| Pipeline (Stack, 4 blocks), P = 2 and 4, GPipe and 1F1B, 8 microbatches: gradient error vs one device | 7.5e-9 in every case |
| Peak activation stashes per stage, P = 4, m = 8 | GPipe 8, 8, 8, 8; 1F1B 4, 3, 2, 1 |
| Ring attention (causal) vs full attention, C = 2 and 4, contiguous and zigzag | max error 4.8e-7 |
| Score blocks per rank in (s/2C)² units, C = 4: contiguous / zigzag | 4, 8, 12, 16 / 9, 9, 9, 9 (max/mean 1.60 / 1.00) |
| Ulysses vs full attention, C = 2 and 4: output and q, k, v gradients | error 0; 4 All-to-Alls forward, 4 backward |
| Forward bytes rank 0 sends, C = 4, s = 64: ring / Ulysses | 12 KiB (3 KV blocks) / 6 KiB (4 All-to-Alls) |
| 512 GPUs as TP 8 x DP 8 x PP 8, 8 per node: groups that leave a node | TP 0 of 64, DP 64 of 64, PP 64 of 64 |
| Same 512 GPUs with TP 16 | all 32 TP groups cross a node boundary |

The sequence-parallel memory saving looks small here because this block materialises attention scores, and the
scores are already split by TP. The part SP removes is the layernorm inputs: 64 KiB each at this size, kept at
1/T. On a real model with fused attention, the replicated regions are a larger share.

### Pipeline schedules on a clock (simulation, no hardware)

Forward 2 units, backward 4 units, transfers free. Full table and timelines in `results/04_pipeline_schedules.txt`.

| P | m | (P-1)/(m+P-1) | GPipe | 1F1B | interleaved v=2 | (P-1)/(vm+P-1) | zero-bubble style | stage-0 stashes: GPipe / 1F1B / interleaved (in full-stage units) |
|---|---|---|---|---|---|---|---|---|
| 4 | 4 | 0.429 | 0.429 | 0.429 | 0.273 | 0.273 | 0.200 | 4 / 4 / 4.0 |
| 4 | 16 | 0.158 | 0.158 | 0.158 | 0.086 | 0.086 | 0.059 | 16 / 4 / 5.5 |
| 8 | 8 | 0.467 | 0.467 | 0.467 | 0.304 | 0.304 | 0.226 | 8 / 8 / 8.0 |
| 8 | 64 | 0.099 | 0.099 | 0.099 | 0.052 | 0.052 | 0.035 | 64 / 8 / 11.5 |

GPipe and 1F1B match the formula at every (P, m) in the table, and so does interleaving with its own formula. The
zero-bubble row comes from a greedy schedule written for this lab (B before F before deferred W, with 1F1B's limit
on forwards in flight). It cuts 1F1B's bubble to a third at these costs. It is not the exact ZB-H1 or ZB-H2 schedule
from Qi et al.

### Collective timings: AMD EPYC 7R13, gloo, 2 ranks

The machine was running other jobs, and repeat runs moved these numbers by a factor of 2. Use them only for the
shape of the curve: a flat latency floor, then time growing with bytes.

| Collective | α (latency floor) | β (fitted bandwidth) |
|---|---|---|
| AllReduce | 2.4 ms | 0.37 GB/s |
| ReduceScatter | 2.3 ms | 0.31 GB/s |
| AllGather | 1.4 ms | 0.63 GB/s |
| send/recv | 0.5 ms | 0.72 GB/s |

### Two GPUs: 2 x NVIDIA H100 80GB HBM3 (NVLink), torch 2.15.0, CUDA 13.2, NCCL

Each job held both GPUs alone, through a lock. Timings are wall-clock through `torch.distributed`, so they include
Python and launch overhead. The latency floor below is therefore higher than what NCCL's own benchmarks report.

**Collectives** (`results/01_collectives_bench_gpu.txt`). Time per call, from 20 back-to-back calls.

| Collective | Floor (4 KB to 16 MB) | 256 MB | Fitted β |
|---|---|---|---|
| AllReduce | 69 to 92 µs | 0.90 ms | 310 GB/s |
| ReduceScatter | 59 to 98 µs (one 229 µs outlier) | 0.58 ms | 485 GB/s |
| AllGather | 59 to 100 µs | 0.55 ms | 535 GB/s |
| send/recv | 56 to 72 µs | 0.75 ms | 369 GB/s |

The time is flat from 4 KB to 16 MB. An AllReduce only starts to pay for bytes above about 20 MB (α·β). Below that
it pays for latency. This is the α–β model from the post: small messages want fusing, which is what DDP buckets do.

**Tensor parallelism, one block, forward + backward, bf16** (`results/08_bench_tp_gpu.txt`). "Fake" replaces each
collective by a local no-op of the right shape, so it times the compute alone. Exposed = real minus fake.

| h, heads, s, b | 1 GPU | TP 2 | TP 2 exposed comm | TP 2 + SP exposed | held activations: 1 GPU / TP 2 / +SP / +SP regather |
|---|---|---|---|---|---|
| 1024, 16, 2048, 4 | 7.63 ms | 4.60 ms | 0.27 ms (5.9%) | 0.73 ms | 772 / 420 / 404 / 388 MB |
| 4096, 32, 2048, 4 | 27.34 ms | 15.52 ms | 1.10 ms (7.1%) | 1.31 ms | 2052 / 1156 / 1092 / 1028 MB |
| 8192, 64, 2048, 2 | 42.48 ms | 22.93 ms | 1.13 ms (4.9%) | 1.03 ms | 2052 / 1156 / 1092 / 1028 MB |

- TP 2 runs one block 1.66 to 1.85 times faster than one GPU. The rest is the communication it cannot hide.
- At h = 4096 each AllReduce carries 64 MB. The collectives table puts a 64 MB AllReduce at 0.26 ms. Four of them
  make 1.02 ms, close to the 1.10 ms exposed. The TP AllReduce sits on the critical path, so nearly all of it shows
  up in the step.
- Sequence parallelism saved 64 MB of held activations at h = 4096, and the regather saved another 64 MB. Each
  layernorm input is s·b·h·2 = 64 MB, so this is two of them kept at 1/2 of their size, then the two gathered
  matmul inputs. At h = 1024 the regather's extra AllGathers cost 3.2 ms, because those messages are small and
  latency-bound.

**Per-rank GEMM width** (same file, part 2). [8192, h] x [h, 4h/T] on one GPU, the up-projection each TP rank runs.

| h | T = 1 | T = 2 | T = 4 | T = 8 | T = 8 / T = 1 |
|---|---|---|---|---|---|
| 1024 | 696 TFLOP/s | 647 | 609 | 444 | 0.64 |
| 4096 | 769 | 732 | 742 | 725 | 0.94 |
| 8192 | 702 | 728 | 724 | 726 | 1.03 |

At h = 1024, TP 8 loses a third of the GEMM throughput. At h = 4096 and 8192 this shape loses almost nothing on an
H100. So "measure the GEMM" cuts both ways: the narrow-GEMM penalty is real for small hidden sizes and small for
large ones at this token count.

**Pipeline, 2 stages, real run** (`results/05_pipeline_bench_gpu.txt`). Stack of 8 blocks, hidden 1024, 16 heads,
global batch 32 x 1024 tokens, bf16. Each stage synchronises the GPU around every op to measure busy time, which
adds some overhead per op.

| m | GPipe step | 1F1B step | measured bubble (GPipe / 1F1B) | (P-1)/(m+P-1) | peak memory, stage 0: GPipe / 1F1B |
|---|---|---|---|---|---|
| 1 | 143 ms | 146 ms | 0.51 / 0.51 | 0.50 | 10840 / 10840 MB |
| 2 | 113 ms | 114 ms | 0.35 / 0.36 | 0.33 | 9564 / 9564 MB |
| 4 | 105 ms | 103 ms | 0.25 / 0.24 | 0.20 | 8932 / 4905 MB |
| 8 | 105 ms | 103 ms | 0.18 / 0.17 | 0.11 | 8628 / 2537 MB |
| 16 | 166 ms | 185 ms | 0.19 / 0.18 | 0.06 | 8500 / 1352 MB |
| 32 | 311 ms | 310 ms | 0.18 / 0.17 | 0.03 | 8484 / 760 MB |

- GPipe and 1F1B take the same time at the same m. 1F1B's peak memory falls with m, because it never holds more
  than 2 stashes. GPipe holds all m. At m = 32 that is 760 MB against 8484 MB.
- The bubble follows the formula while microbatches are large. Past m = 8 each microbatch is 1 to 2 sequences.
  Busy time then grows from 71 ms to over 250 ms, and the step gets slower even though the formula's bubble keeps
  shrinking. This is the post's three-way trade between bubble, kernel efficiency and memory. Part of the growth is
  this executor's per-op synchronisation, so the exact turning point is specific to this code.

**Ring attention, 2 GPUs** (`results/06_ring_bench_C2.txt`). Causal, zigzag, 16 heads of 128, bf16 inputs, scores
in fp32 without a fused kernel.

| seq | posting the KV exchange early | waiting for it first |
|---|---|---|
| 8192 | 12.1 ms | 12.4 ms |
| 16384 | 42.4 ms | 42.0 ms |
| 32768 | 156.4 ms | 157.7 ms |

Overlap makes no measurable difference here. With C = 2 there is one rotation. At seq 32768 it moves 128 MB, which
the collectives table puts near 0.4 ms against 156 ms of attention compute. That is the post's condition for the
exchange to hide: many tokens per rank. With C = 2 on NVLink there is nothing left to hide.

## What was not measured

- DDP, ZeRO and FSDP. C5 builds and measures them.
- Expert parallelism, distributed checkpointing and the failure modes. The post explains them; this lab builds none.
- Anything beyond 2 GPUs or across nodes. The TP node-boundary cliff, HSDP, interleaved and zero-bubble schedules on
  real hardware, and the straggler tax are not measured. The mesh script only does the placement arithmetic.
- The ring-attention backward pass. Ring attention here is forward only. Ulysses has forward and backward.
- The zero-bubble row is a greedy schedule on a simulated clock, not the paper's ZB-H1 or ZB-H2 and not a real run.
- Paper-scale numbers. The post quotes Megatron-scale configurations (for example 512 GPUs); none of them were run.
- Dropout. The blocks have none, so the SP saving on dropout masks does not appear.
