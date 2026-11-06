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
tests. See `results/tests.txt`.

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

These need an accelerator, and this is the only one measured so far. The laptop was shared with other jobs.

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
| AMD EPYC 7R13, 8 threads | Every other file |

`07_break_it.txt` shows the device-mismatch error on the `meta` device, because the EPYC runs had no accelerator.
`07_break_it_apple_m5.txt` shows the same mistake on MPS.

### CUDA: not yet measured

No CUDA numbers exist yet. These commands measure them. `pick_device` selects CUDA first and `synchronize` calls
`torch.cuda.synchronize`.

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
