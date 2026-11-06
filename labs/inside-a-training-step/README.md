# Lab: Inside a Training Step

Code companion for the post [Inside a Training Step](https://srini-anirudh.github.io/blogs/inside-a-training-step/).

The post opens up one training step of a Llama-2-7B-shaped model. It tracks what arithmetic is done, what memory
is allocated, what data moves, and what depends on what. This lab does two things with that.

First, it recomputes every back-of-envelope number in the post from the six architecture numbers. Each one is
printed next to the post's figure.

Second, it makes the mechanisms happen on real models:

- Backward costs twice the forward. PyTorch's FLOP counter shows one GEMM in and two out.
- Activation memory is a mountain. A saved-tensor tracker watches it rise through the forward, peak at the loss
  and drain through the backward, with nested lifetimes per block.
- Precision has four parts. A BF16 accumulator loses a long sum. A BF16 weight ignores small Adam updates. FP16
  flushes small gradients to zero.
- Checkpointing trades FLOPs for memory, both full and selective.
- The optimizer step is bandwidth.
- On an H100 at the post's layer shape, the lab times each op of one block. It also measures the allocator's
  memory ledger, the memory curve, MFU and the memory wall.

The small models come from C4's `TinyLLM` ([`c04-llm/tinyllm`](../../c04-llm)), imported rather than rebuilt.
Gradient accumulation and DDP live in C5 (see the table below), so this lab does not repeat them.

## Layout

```
labs/inside-a-training-step/
├── stepanatomy/
│   ├── accounting.py   # the post's arithmetic: params, per-op FLOPs and bytes, ledger, 6ND, optimizer bytes
│   ├── models.py       # C4's TinyLLM at a given shape (MHA, untied head, as in Llama 2)
│   ├── flops.py        # forward and backward FLOPs from torch.utils.flop_counter
│   ├── lifetime.py     # saved-tensor tracker: the activation mountain and per-block lifetimes
│   ├── recompute.py    # full and selective activation checkpointing
│   ├── mixed.py        # MasterWeightAdam: BF16 model, FP32 master and moments; the 16-byte ledger
│   ├── precision.py    # BF16 accumulation, vanishing updates, FP16 range
│   ├── optbench.py     # Adam step time turned into bytes per second
│   ├── gpustep.py      # CUDA only: op timings, allocator ledger, step time and MFU, memory wall, H2D
│   └── report.py       # writes results/ with a hardware line and the command
├── scripts/            # 01 to 08, listed below
├── tests/
└── results/            # real outputs, one .txt (and .json) per script
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Nothing is downloaded. Set `OMP_NUM_THREADS` on a
shared machine; the CPU numbers below used 8 threads.

## Run everything

```bash
# Tests: offline CPU, then CUDA.
python -m pytest labs/inside-a-training-step -m "not slow and not gpu and not network"
python -m pytest labs/inside-a-training-step -m gpu

# CPU scripts. Each takes seconds.
python labs/inside-a-training-step/scripts/01_accounting.py
python labs/inside-a-training-step/scripts/02_flops.py
python labs/inside-a-training-step/scripts/03_lifetime.py
python labs/inside-a-training-step/scripts/04_precision.py
python labs/inside-a-training-step/scripts/05_checkpointing.py
python labs/inside-a-training-step/scripts/06_ledger.py
python labs/inside-a-training-step/scripts/07_optimizer.py

# GPU scripts. A few minutes on one H100.
python labs/inside-a-training-step/scripts/07_optimizer.py --device cuda --params 1.07e9 --hidden 4096
python labs/inside-a-training-step/scripts/08_gpu_step.py --layers 8
```

## Post sections and scripts

| Post section | Script | Module |
|---|---|---|
| Five lines that hide everything (the 6.74B model) | `01_accounting.py` | `accounting.py` |
| Before the GPU: a batch's journey | `08_gpu_step.py`, part 4 | `gpustep.h2d` |
| Inside one transformer block | `01_accounting.py`, `08_gpu_step.py` part 1 | `accounting.block_ops`, `gpustep.op_table` |
| Reading shapes off an architecture diagram | `01_accounting.py` | `accounting.tensor_sizes` |
| Why compute scales like N × D | `01_accounting.py`, `02_flops.py` | `flops.py` |
| Backprop: one GEMM in, two out | `02_flops.py` | `flops.linear_flops` |
| The graph and the activation lifetime | `03_lifetime.py` | `lifetime.py` |
| The training memory ledger | `01_accounting.py`, `06_ledger.py`, `08_gpu_step.py` parts 2 and 3 | `mixed.py`, `gpustep.py` |
| Four precisions, not one | `04_precision.py` | `precision.py` |
| Activation checkpointing | `02_flops.py`, `05_checkpointing.py`, `08_gpu_step.py` part 2 | `recompute.py` |
| Gradient accumulation | not rebuilt: C5's [`check_accumulation.py`](../../c05-pretraining/scripts/check_accumulation.py) | C5 `pretrain/step.py` |
| The optimizer step is bandwidth | `01_accounting.py`, `07_optimizer.py` | `optbench.py` |
| Utilization is not efficiency (MFU) | `08_gpu_step.py` part 2 | `gpustep.step_report` |
| Two walls, one question | `01_accounting.py`, `08_gpu_step.py` part 3 | `gpustep.memory_wall` |

## Results

Every number names its hardware. Raw outputs are in `results/`, one file per script, each starting with the
hardware and the command.

### AMD EPYC 7R13, 8 threads, torch 2.15.0 (CPU, shared machine)

Counts (FLOPs, bytes, ratios) are exact and do not depend on load. Timings are medians on a loaded machine
(load average near 50 on 192 CPUs), so treat them as orders of magnitude.

| Measurement | Result | File |
|---|---|---|
| Parameters of the post's model, 32(4H² + 3HI) + 2VH | 6,738,149,376 (post: 6.74e9) | `01_accounting.txt` |
| Projections' share of a block's FLOPs (causal) | 92.3% (post: 92%) | `01_accounting.txt` |
| Norms, adds and SiLU-multiply share of FLOPs | 0.022% (post: under 0.05%) | `01_accounting.txt` |
| SiLU-multiply HBM traffic, per layer / per step | 271 MB / 8.66 GB (post: 270 MB, 8.6 GB) | `01_accounting.txt` |
| Attention crossover S* = P_layer / 2H | 24,704 tokens (post: about 24,700) | `01_accounting.txt` |
| 6ND for 2T tokens, years on one H100 at 45% MFU | 8.09e22 FLOPs, 5.76 years (post: 8.1e22, about six) | `01_accounting.txt` |
| Model states, 16 bytes per parameter | 107.8 GB, 35% over 80 GB (post: 108 GB, 35%) | `01_accounting.txt` |
| Activations, L·B·S·16H·2 bytes | 17.2 GB; full checkpointing keeps 1.61 GB, a 10.7x cut (post: about 10x) | `01_accounting.txt` |
| Adam step: bytes, FLOPs, traffic, time at 2.5 TB/s | 28 B, 12 FLOP, 189 GB, 75 ms (post: same) | `01_accounting.txt` |
| Backward / forward FLOPs for one GEMM | 2.0 (1.0 if W is frozen, 1.0 if dX is skipped) | `02_flops.txt` |
| TinyLLM backward / forward FLOPs | 2.000 (naive attention) | `02_flops.txt` |
| TinyLLM forward FLOPs / 2ND, measured vs 1 + 2LSH/N, S = 64 to 2048 | equal to 4 digits (1.04, 1.159, 1.637, 2.274) | `02_flops.txt` |
| Backward FLOPs with one middle block frozen / bottom 2 blocks and embeddings frozen | 0.897 / 0.517 of the trainable model | `02_flops.txt` |
| Training FLOPs / ND: none / selective / full checkpointing | 6.955 / 6.955 / 8.766 (x1.26) | `02_flops.txt` |
| Saved activations at the turn, TinyLLM L=4, H=256, B=2, S=256, FP32 | 44.3 MiB; peak coincides with the loss; 0 bytes live after backward | `03_lifetime.txt` |
| Saved elements per token per block (SDPA) | 20.9 H (post's count: 16H; see the note below) | `03_lifetime.txt` |
| Saved bytes per block, naive / SDPA, S = 128 to 1024 | ratio 1.10, 1.20, 1.41, 1.82 | `03_lifetime.txt` |
| Relative error of a 4096-term BF16 dot product: BF16 / FP32 accumulator | 0.748 / 8.4e-5 | `04_precision.txt` |
| BF16 sum of 4096 ones | 256 | `04_precision.txt` |
| 1000 updates of -1e-4·w to a weight of 1.0: BF16 / FP32 master | 1.0 / 0.9048 (exact 0.9048) | `04_precision.txt` |
| Adam fit, loss after 600 steps: FP32 / BF16 weights + FP32 master / pure BF16 | 5.1e-12 / 7.6e-4 / 6.7e-3 (never moved) | `04_precision.txt` |
| Gradient-sized values flushed to zero: FP16 / FP16 with loss scale 2^16 / BF16 | 31% / 0% / 0% | `04_precision.txt` |
| Checkpointing, saved at the turn: full / selective | 9.3x / 1.27x less than none; gradients identical (max diff 0) | `05_checkpointing.txt` |
| Checkpointing, step time: full / selective | 1.28x / 1.11x (rough) | `05_checkpointing.txt` |
| Ledger of a BF16 TinyLLM with MasterWeightAdam | 16.0 bytes per parameter | `06_ledger.txt` |
| AdamW step, 50.6M params: for-loop / foreach / fused / MasterWeightAdam | 72.6 / 81.7 / 18.0 / 67.6 ms | `07_optimizer_cpu.txt` |

Notes on what the numbers say:

- The post's 16H count is for a kernel that saves one tensor per fused op. C4's TinyLLM is unfused: each RMSNorm
  saves its pre-gain value and the SiLU-multiply saves `silu(gate)` as a fourth I-wide tensor. That is 10 + 4·I/H =
  20.75H for this shape, and the tracker measures 20.9H. The `selective` mode removes exactly those tensors, and a
  test checks that it lands near 8 + 3·I/H.
- Full checkpointing here recomputes 1.26x the FLOPs, not 1.33x. The LM head is not checkpointed, and torch's
  checkpoint does not rerun a block's last GEMM (`down_proj`), because nothing saves its output. A test checks
  this FLOP for FLOP.
- Full checkpointing cuts what is saved at the turn by 9.3x. This tracker does not see the one block that is
  rebuilt during backward, so it is not a peak-memory measurement. The GPU script measures peak memory.
- On this CPU the `foreach` AdamW is slower than the for-loop. The `fused` CPU kernel is 4x faster than both.
  The post's claim about fused and multi-tensor optimizers is about GPU launch overhead and HBM traffic, which a CPU does not show.
- The pure-BF16 run is a worst case, with lr 3e-4 and weights near 1.0. It shows the mechanism, not how every
  BF16 run behaves.

### NVIDIA H100 80GB HBM3: not yet measured

The GPU jobs were queued behind other jobs when this README was written, and none had run. These commands
produce the numbers. They write `results/08_gpu_step.txt`, `results/07_optimizer_gpu.txt` and the GPU test run.

```bash
python -m pytest labs/inside-a-training-step -m gpu
python labs/inside-a-training-step/scripts/07_optimizer.py --device cuda --params 1.07e9 --hidden 4096
python labs/inside-a-training-step/scripts/08_gpu_step.py --layers 8
```

`08_gpu_step.py` measures, at the post's layer shape (H = 4096, I = 11008, V = 32000, S = 4096, B = 1):
the time of each op of one block, the allocator's bytes per parameter, the memory curve at each block, forward,
backward and optimizer times, MFU with and without full checkpointing, whether the 6.7B model's states fit on
one card, and pageable against pinned host-to-device copies. Its code has run on CPU only through the
accounting helpers. The CUDA paths have never run, so expect to fix a bug or two on the first run.

## What was not measured

- Everything at the post's scale on CPU. The CPU scripts use a 4-layer, 256-wide model. The post's 7B numbers
  come from arithmetic (`01_accounting.py`), not from running a 7B model.
- Any H100 number, until the GPU jobs above run.
- Data loading, worker processes and double buffering. Only the host-to-device copy is in `08_gpu_step.py`.
- FP8, selective checkpointing's 4% overhead (Korthikanti et al.) and ZeRO. The lab shows the mechanism of
  selective recompute on one small model. It does not reproduce the paper's figure.
- Gradient accumulation and `no_sync()`. See C5 (`check_accumulation.py`, `ddp_equivalence.py`).
- `nvidia-smi` utilization against MFU. The post's claim is definitional.
