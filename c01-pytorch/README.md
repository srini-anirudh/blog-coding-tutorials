# C1: PyTorch From First Principles

A small neural-network training project written in raw PyTorch. The package `scratchnet` builds tensor inspection
tools, matrix multiplication, a `Linear` layer, an `MLP`, MSE and cross entropy, SGD and AdamW, datasets with collate
functions, a `DataLoader`, the full training loop and debugging hooks. Each piece is checked against the PyTorch
implementation it replaces.

Tutorial: https://srini-anirudh.github.io/blogs/pytorch-from-first-principles/

By the end you can explain exactly what happens between `loss.backward()` and `optimizer.step()`.

## Layout

```
c01-pytorch/
├── scratchnet/
│   ├── tensors.py         # storage, strides, offsets, manual view/transpose/permute
│   ├── broadcasting.py    # the broadcasting rule and stride-0 expansion
│   ├── device.py          # pick_device, synchronize
│   ├── matmul.py          # matmul from Python loops to mm, bmm and broadcast matmul
│   ├── autograd_tools.py  # walk the grad_fn graph
│   ├── layers.py          # Linear, ReLU, Dropout, Standardize, MLP
│   ├── losses.py          # MSE, log_softmax, cross entropy (log_softmax + gather)
│   ├── optim.py           # SGD, AdamW, grad_norm, clip_grad_norm
│   ├── data.py            # SpiralDataset, VarLenDataset, collate_dicts, pad_collate, SimpleDataLoader
│   ├── train.py           # TrainConfig, train_step, evaluate, fit, checkpoints
│   └── debug.py           # ActivationStats, GradNormMonitor, NaNDetector, check_same_device
├── scripts/               # runnable demos, listed below
├── benchmarks/            # transfer, matmul, optimizers, dataloader
├── configs/spiral_adamw.yaml
├── results/               # captured outputs of every script and benchmark
└── tests/
```

## Run

From the repository root, after `pip install -e ".[dev]"`. PyTorch uses every core by default, so set
`OMP_NUM_THREADS` on a shared machine. The CPU numbers below used `OMP_NUM_THREADS=8`.

```bash
# Tests.
python -m pytest c01-pytorch -m "not slow and not gpu and not network"   # CPU only
python -m pytest c01-pytorch -m gpu                                      # needs CUDA; skipped without it

# Demos.
python c01-pytorch/scripts/01_tensors.py
python c01-pytorch/scripts/02_broadcasting.py
python c01-pytorch/scripts/03_autograd.py
python c01-pytorch/scripts/04_losses.py
python c01-pytorch/scripts/05_train.py --config c01-pytorch/configs/spiral_adamw.yaml
python c01-pytorch/scripts/05_train.py --optimizer sgd
python c01-pytorch/scripts/05_train.py --compare          # manual vs torch optimizers
python c01-pytorch/scripts/06_backward_to_step.py         # one step, every piece of state printed
python c01-pytorch/scripts/07_break_it.py                 # or one of: lr zero_grad eval shape device
python c01-pytorch/scripts/08_linear_manual.py
python c01-pytorch/scripts/09_modules.py
python c01-pytorch/scripts/10_data.py
python c01-pytorch/scripts/11_matmul.py

# Benchmarks. They print tables.
python c01-pytorch/benchmarks/transfer.py                # needs an accelerator; otherwise it says so and exits
python c01-pytorch/benchmarks/matmul.py
python c01-pytorch/benchmarks/optimizers.py
python c01-pytorch/benchmarks/dataloader.py
```

Which script covers which section of the tutorial:

| Section | Script or benchmark |
|---|---|
| Tensors, storage, strides and views | `01_tensors.py` |
| Broadcasting | `02_broadcasting.py` |
| CPU and GPU tensors | `benchmarks/transfer.py` |
| Matrix multiplication | `11_matmul.py`, `benchmarks/matmul.py` |
| Autograd | `03_autograd.py` |
| A linear layer by hand, vanilla SGD | `08_linear_manual.py` |
| `nn.Module` | `09_modules.py` |
| Loss functions | `04_losses.py` |
| Optimizers | `05_train.py --compare`, `benchmarks/optimizers.py` |
| Dataset and DataLoader | `10_data.py`, `benchmarks/dataloader.py` |
| The full training loop | `05_train.py`, `06_backward_to_step.py` |
| Debugging | `07_break_it.py` |

## Results

Every number names its hardware. Raw output is in `results/`, under the name of the script that wrote it.

Tests on AMD EPYC 7R13, 8 threads, torch 2.15.0: 50 passed, 6 `gpu` tests deselected. There are no `network` or `slow`
tests. See `results/tests.txt`. On an NVIDIA H100 80GB HBM3 (torch 2.15.0+cu132, CUDA 13.2) the 6 `gpu` tests passed.
See `results/cuda/tests_gpu.txt`.

### AMD EPYC 7R13, 8 threads, torch 2.15.0

Timings are medians. The machine was running other jobs (load average near 54 on 192 CPUs), so treat timings as orders of
magnitude. In a repeat run the 1024 x 1024 `torch.mm` moved from 8.0 ms to 3.4 ms. Losses and accuracies do not depend
on load.

| Measurement | Result |
|---|---|
| Spiral MLP, AdamW, lr 0.01, 60 epochs | train loss 0.0015, val accuracy 0.998 |
| Spiral MLP, SGD + momentum, lr 0.1, 60 epochs | train loss 0.0018, val accuracy 0.997 |
| `scratchnet.AdamW` vs `torch.optim.AdamW`, largest epoch-loss gap over 60 epochs | 0.0 |
| `scratchnet.SGD` vs `torch.optim.SGD`, largest epoch-loss gap over 60 epochs | 0.0 |
| `optimizer.step()`, 6.3 M parameters: ours / torch for-loop / foreach / fused | 5.4 / 7.6 / 18.1 / 2.4 ms |
| Matmul speed: `mm_loops` (n=32) / `mm_rows` (n=256) / `torch.mm` (n=2048) | 0.0004 / 3.1 / 635 GFLOP/s |
| 2048 x 2048 `torch.mm` | 27.1 ms |
| DataLoader, 2 ms per example, epoch 2: 0 workers / 2 persistent / 4 persistent | 496 / 922 / 1778 examples/s |
| `num_workers=4`, new workers each epoch, epoch 2: torch / `SimpleDataLoader` | 1089 / 1508 examples/s |
| lr 10 with plain SGD: first non-finite output | step 24, in `blocks.1.0` |
| lr 10 with clipping at 1.0, 30 epochs: does it recover? | no: train loss 9.80, val accuracy 0.333 |
| Forgetting `zero_grad`, 30 epochs: loss with / without the call | 0.0136 / 351.7 |

The `SimpleDataLoader` comparison is one run. Two earlier runs measured `SimpleDataLoader` at 1414 and 1067 examples/s,
so it does not show that either loader is faster.

### Apple M5, 10 cores, 16 GB, torch 2.15.0, 2 CPU threads

These need an accelerator. The laptop was shared with other jobs. The CUDA numbers are further down.

| Measurement | Result |
|---|---|
| `optimizer.step()`, 6.3 M parameters, MPS: ours / torch for-loop / foreach / fused | 8.6 / 11.2 / 10.9 / 5.5 ms |
| 2048 x 2048 `torch.mm`: CPU, 2 threads / MPS | 35.0 / 6.0 ms |
| 4096 x 4096 `torch.mm` on MPS, without / with `synchronize()` | 0.285 / 53.6 ms |
| Spiral MLP, AdamW, 20 epochs on MPS | 17.59 s, val accuracy 0.980 |
| 256 MB copy to MPS, pageable: run 1 / run 2 | 15.2 / 8.1 GB/s |
| 256 MB copy to MPS, pinned: run 1 / run 2 | 21.7 / 47.6 GB/s |
| 256 MB copy back to CPU: run 1 / run 2 | 14.0 / 4.7 GB/s |

The two transfer runs disagree by a factor of two or more, and in run 2 one 64 MB copy took 41 s to finish. Do not read a
pinned speed-up into these numbers.

### Which machine wrote each file

| Machine | Files in `results/` |
|---|---|
| Apple M5 | `05_train_mps.txt`, `transfer.txt`, `transfer.json`, `transfer_run2.txt`, `transfer_run2.json`, `matmul_apple_m5.txt`, `optimizers_apple_m5.txt`, `07_break_it_apple_m5.txt` |
| NVIDIA H100 80GB HBM3, host AMD EPYC 7R13, 8 threads | Everything in `results/cuda/` |
| AMD EPYC 7R13, 8 threads | Every other file |

`07_break_it.txt` shows the device-mismatch error on the `meta` device, because the EPYC runs had no accelerator.
`07_break_it_apple_m5.txt` shows the same mistake on MPS, and `cuda/07_break_it_device.txt` on CUDA.

### CUDA: NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2

One H100, held alone for the whole run. The host CPU is also an AMD EPYC 7R13 with `OMP_NUM_THREADS=8`, so the CPU
columns below come from that host in the same run. Other jobs ran on the node's other GPUs. Raw output is in
`results/cuda/`. `pick_device` selects CUDA first and `synchronize` calls `torch.cuda.synchronize`.

The 6 `gpu` tests passed (`results/cuda/tests_gpu.txt`). They cover `.cuda()` vs `.to()`, a pinned `non_blocking` copy,
the real device mismatch, `scratchnet.AdamW` vs `torch.optim.AdamW` on CUDA, resuming from a checkpoint on CUDA, and
`SimpleDataLoader` pinning.

| Measurement | Result |
|---|---|
| Spiral MLP, AdamW, lr 0.01, 60 epochs on CUDA | train loss 0.0015, val accuracy 0.998, 4.91 s |
| The same run on the EPYC CPU (`05_train_adamw.txt`, other machine, loaded) | train loss 0.0015, val accuracy 0.998, 5.79 s |
| `optimizer.step()`, 6.3 M parameters, CUDA: ours / torch for-loop / foreach / fused | 3.24 / 3.78 / 0.55 / 0.23 ms |
| The same on the host CPU, 8 threads | 4.11 / 4.34 / 10.51 / 0.96 ms |
| `torch.mm` on CUDA, n = 256 / 1024 / 2048 / 4096 | 0.049 / 0.085 / 0.369 / 2.695 ms |
| The same as GFLOP/s | 687 / 25,228 / 46,606 / 51,001 |
| 2048 x 2048 `torch.mm`: host CPU, 8 threads / CUDA | 22.7 / 0.369 ms |
| 4096 x 4096 `torch.mm` on CUDA, without / with `synchronize()` | 0.037 / 2.695 ms |
| 256 MB copy to the GPU: pageable / pinned | 11.7 / 14.3 GB/s |
| 256 MB copy back to the CPU | 1.35 GB/s |
| 64 MB pinned copy, `non_blocking=True`: call returns / copy finishes | 0.036 / 4.74 ms |
| DataLoader, 4 persistent workers, epoch 2: without / with `pin_memory` | 1813 / 1771 examples/s |

The spiral MLP is too small to keep a GPU busy, so CUDA trains it barely faster than the CPU. The same seed gives the
same loss and accuracy on both. The matmul runs in float32 with the default precision (no TF32). Without
`synchronize()`, the 4096 matmul seems to take 0.037 ms: that is only the time to queue it.

Pinned copies were faster at every size, from 1.8 to 3.1 GB/s at 64 KB and from 11.7 to 14.3 GB/s at 256 MB. These
copies ran well below what the H100's PCIe Gen5 link can carry. Jobs on the other GPUs shared the host, and this run
does not show which limit applied. Copies back to the CPU land in fresh pageable memory. They were about ten times
slower than pinned copies to the GPU. Pinning did not change
DataLoader throughput here, because each batch is a few KB and the 2 ms of CPU work per example dominates. This is one
run of each benchmark.

`results/cuda/07_break_it_device.txt` shows the real device-mismatch error: `cuda:0 and cpu`.

To measure them again:

```bash
python -m pytest c01-pytorch -m gpu
python c01-pytorch/benchmarks/transfer.py --device cuda
python c01-pytorch/benchmarks/matmul.py
python c01-pytorch/benchmarks/optimizers.py
python c01-pytorch/benchmarks/dataloader.py               # adds a pin_memory row
python c01-pytorch/scripts/05_train.py --optimizer adamw --device cuda
python c01-pytorch/scripts/07_break_it.py device
```

## Theory

- [What Autograd Actually Does](https://srini-anirudh.github.io/blogs/what-autograd-actually-does/)
- [What an Optimizer Actually Does](https://srini-anirudh.github.io/blogs/what-an-optimizer-actually-does/)
