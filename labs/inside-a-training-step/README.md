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

### NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2

One GPU per job, so no timing shared the card with another job. `08_gpu_step.py` takes about a minute. Its
step runs a BF16 TinyLLM (C4) with 8 layers at the post's layer shape (H = 4096, I = 11008, V = 32000,
S = 4096, B = 1): 1.88B parameters, not the post's 6.74B. Medians of 5 steps (20 for single ops).

| Measurement | Result | File |
|---|---|---|
| One block, forward, op by op (eager) | 3.37 ms: projections 68% of the time, attention 9%, norms + adds + SiLU-multiply 23% | `08_gpu_step.txt` |
| Same ops' share of FLOPs | projections 92%, norms + adds + SiLU-multiply 0.022% | `01_accounting.txt` |
| Projection GEMMs | 667 to 757 TFLOP/s, 67 to 77% of the 989 TFLOP/s dense peak | `08_gpu_step.txt` |
| Causal flash attention, S = 4096 | 470 TFLOP/s (counting the causal half) | `08_gpu_step.txt` |
| Eager RMSNorm / SiLU-multiply | 0.272 ms (247 GB/s) / 0.165 ms (1.64 TB/s) | `08_gpu_step.txt` |
| Same two ops under `torch.compile` | 0.064 ms (4.2x faster) / 0.128 ms (1.3x faster) | `08_gpu_step.txt` |
| Allocator bytes per parameter: weights / grads / master + m + v | 2.00 / 2.00 / 12.00 (16 in all) | `08_gpu_step.txt` |
| Memory at the start of the step / at the loss / after backward | 24.53 / 30.84 / 28.03 GiB | `08_gpu_step.txt` |
| Peak allocated, no checkpointing / full checkpointing | 31.81 / 29.50 GiB | `08_gpu_step.txt` |
| Forward / backward / optimizer, no checkpointing | 31.9 / 65.9 (2.07x) / 59.0 ms; step 156.7 ms | `08_gpu_step.txt` |
| Forward / backward / optimizer, full checkpointing | 32.1 / 91.8 (2.86x) / 58.9 ms; step 182.9 ms | `08_gpu_step.txt` |
| Forward + backward time with full checkpointing | 1.27x (FLOP count: 1.26x) | `08_gpu_step.txt` |
| MFU, 6ND over the whole step / forward + backward only (no checkpointing) | 27.7% / 44.5% | `08_gpu_step.txt` |
| MFU, same two with full checkpointing | 23.8% / 35.1% | `08_gpu_step.txt` |
| 6.74B model's states on one card, one category at a time | fits through Adam m (75.3 GiB), fails at Adam v; the ledger needs 100.4 GiB of 79.2 GiB | `08_gpu_step.txt` |
| 256 MiB host-to-device copy, pageable / pinned | 12.4 / 14.3 GB/s | `08_gpu_step.txt` |
| A GEMM plus a copy on a side stream: pageable / pinned, against the copy alone | 23.1 vs 21.1 ms / 18.8 vs 18.8 ms | `08_gpu_step.txt` |
| AdamW step, 1.01B params: for-loop / foreach / fused / MasterWeightAdam | 27.7 / 27.8 / 9.8 / 31.8 ms | `07_optimizer_gpu.txt` |
| Useful rate at 28 bytes per parameter: fused / for-loop | 2.88 / 1.03 TB/s | `07_optimizer_gpu.txt` |
| The 6.74B step at the fused rate | 66 ms (post: about 75 ms at 2.5 TB/s) | `07_optimizer_gpu.txt` |

What these numbers say about the post:

- The block's cost is where the post puts it. The GEMMs run at 67 to 77% of peak, so they are compute-bound.
  The norms, adds and SiLU-multiply are 0.022% of the FLOPs but 23% of the block's time, because eager PyTorch
  runs each as several passes over HBM. `torch.compile` fuses RMSNorm and wins 4.2x. This is the post's fusion
  argument, measured per op. This lab did not measure the post's end-to-end 10 to 20% from `torch.compile`.
- The memory curve is the mountain. Memory climbs 0.71 GiB per block through the forward, peaks at the loss
  and drains in the backward. The peak, 31.81 GiB, is 0.97 GiB above the memory at the loss. It is a transient
  in the head's backward. This lab did not locate it, but it fits the post's logits and OOM-in-the-backward point.
- Activations per layer are larger than the post's count. 0.71 GiB per layer is about 22.7H BF16-sized
  elements per token, against the post's 16H, because C4's blocks are unfused (see the CPU notes above). For 8 layers
  the post's formula gives 4.00 GiB and the blocks hold 5.76 GiB. The remaining 0.55 GiB at the loss is the head.
- Full checkpointing here saves less than the post's "about 10x". The peak fell by 2.3 GiB (7%), and the
  non-weight part of the peak, 7.28 GiB with the 3.5 GiB of gradients inside it, fell to 4.97 GiB. Two reasons:
  the model has only 8 layers, and the logits and loss are never checkpointed. The post's 10x counts the layers' activations alone,
  at 32 layers. The compute side matches: +27% time, not +33%.
- MFU is 44.5% for forward + backward and 27.7% for the whole step. The gap is the optimizer: 59 ms, 38% of
  the step, against the post's 20% at 6.74B. `MasterWeightAdam` is eager and unfused. It moves 0.89 TB/s
  against 2.88 TB/s for the fused kernel. The post lists "an optimizer that isn't fused" among the causes of an MFU below 30%, and here it is.
  Scaling the fused rate to 6.74B gives 66 ms, in line with the post's 75 ms.
- The memory wall is real on this card. States alone need 100.4 GiB (108 GB) and the card has 79.2 GiB.
- Pinned memory overlaps the copy with compute and pageable memory does not. The GEMM added nothing to the
  pinned copy (18.8 ms either way), and a pageable copy plus a GEMM took the sum of the two. The bandwidth gain
  was small, 14.3 against 12.4 GB/s, and both are far below a modern PCIe link. This lab did not find out why. Other
  jobs may share the host. Do not read a link speed from these numbers.

Two things in the GPU path had to be fixed after the first run, and they matter if you extend it:

- `memory_allocated()` counts RoPE's cos/sin tables, which are buffers and are cast to BF16 with the weights.
  At the small test shape they add 0.04 bytes per parameter (2.04 in all). At 1.88B parameters they are invisible.
  The test now checks `parameter bytes + buffer bytes`.
- The allocator hands out a whole cached block when the leftover would be under 1 MiB, and `memory_allocated()` counts
  the block. That made the optimizer state read 12.16 bytes per parameter at the small shape, and building the
  model under `with torch.device("cuda")` left 0.56 MB live that no tensor accounted for. I dropped the redundant
  `.clone()` in `MasterWeightAdam` and the device context, and the test allows up to 2 MiB of slack. This
  is the explanation that fits the measurements, and I did not confirm it in the allocator's source.

```bash
python -m pytest labs/inside-a-training-step -m gpu       # 3 passed
python labs/inside-a-training-step/scripts/07_optimizer.py --device cuda --params 1.07e9 --hidden 4096
python labs/inside-a-training-step/scripts/08_gpu_step.py --layers 8
```

## What was not measured

- Anything at the post's scale on CPU. The CPU scripts use a 4-layer, 256-wide model. The post's 7B numbers
  come from arithmetic (`01_accounting.py`), not from running a 7B model.
- The post's 6.74B model on a GPU. The GPU step uses 8 layers (1.88B parameters), so its absolute times, memory
  and MFU are not the post's 32-layer numbers. The 6.74B figures here come from arithmetic, or from scaling a rate.
- A multi-GPU or sharded step (the next post).
- Data loading, worker processes and double buffering. Only the host-to-device copy is in `08_gpu_step.py`.
- FP8, selective checkpointing's 4% overhead (Korthikanti et al.) and ZeRO. The lab shows the mechanism of
  selective recompute on one small model. It does not reproduce the paper's figure.
- Gradient accumulation and `no_sync()`. See C5 (`check_accumulation.py`, `ddp_equivalence.py`).
- `nvidia-smi` utilization against MFU. The post's claim is definitional.
