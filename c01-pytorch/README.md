# C1: PyTorch From First Principles

A small neural-network training project written in raw PyTorch. The package `scratchnet` builds tensor inspection
tools, a `Linear` layer, an `MLP`, MSE and cross entropy, SGD and AdamW, datasets with collate functions, the full
training loop and debugging hooks. Each piece is checked against the PyTorch implementation it replaces.

Tutorial: https://srini-anirudh.github.io/blogs/pytorch-from-first-principles/

By the end you can explain exactly what happens between `loss.backward()` and `optimizer.step()`.

## Layout

```
c01-pytorch/
├── scratchnet/
│   ├── tensors.py         # storage, strides, offsets, manual view/transpose/permute
│   ├── broadcasting.py    # the broadcasting rule and stride-0 expansion
│   ├── device.py          # pick_device, synchronize
│   ├── matmul.py          # matmul from Python loops to torch.mm and torch.bmm
│   ├── autograd_tools.py  # walk the grad_fn graph
│   ├── layers.py          # Linear, ReLU, Dropout, Standardize, MLP
│   ├── losses.py          # MSE, log_softmax, cross entropy (log_softmax + gather)
│   ├── optim.py           # SGD, AdamW, grad_norm, clip_grad_norm
│   ├── data.py            # SpiralDataset, VarLenDataset, collate_dicts, pad_collate
│   ├── train.py           # TrainConfig, train_step, evaluate, fit, checkpoints
│   └── debug.py           # ActivationStats, GradNormMonitor, NaNDetector, check_same_device
├── scripts/               # one runnable demo per section of the tutorial
├── benchmarks/            # transfer, matmul, optimizers, dataloader
├── configs/spiral_adamw.yaml
├── results/               # captured outputs of every script and benchmark
└── tests/
```

## Run

From the repository root, after `pip install -e ".[dev]"`:

```bash
# Tests. CPU only; one assertion also uses MPS when it is available.
python -m pytest c01-pytorch -m "not slow and not gpu"

# Demos, in tutorial order.
python c01-pytorch/scripts/01_tensors.py
python c01-pytorch/scripts/02_broadcasting.py
python c01-pytorch/scripts/03_autograd.py
python c01-pytorch/scripts/04_losses.py
python c01-pytorch/scripts/05_train.py --config c01-pytorch/configs/spiral_adamw.yaml
python c01-pytorch/scripts/05_train.py --compare          # manual vs torch optimizers
python c01-pytorch/scripts/06_backward_to_step.py         # one step, every piece of state printed
python c01-pytorch/scripts/07_break_it.py                 # lr | zero_grad | eval | shape | device

# Benchmarks. They print tables.
python c01-pytorch/benchmarks/transfer.py      # needs MPS or CUDA
python c01-pytorch/benchmarks/matmul.py
python c01-pytorch/benchmarks/optimizers.py
python c01-pytorch/benchmarks/dataloader.py
```

## Results

Hardware: Apple M5, 10 cores, 16 GB, torch 2.15.0, `torch.set_num_threads(2)`. Medians. This machine was shared with
other jobs, so treat the numbers as ratios and orders of magnitude. Raw output is in `results/`.

Tests: 42 passed.

| Measurement | Result |
|---|---|
| Spiral MLP, AdamW, lr 0.01, 60 epochs, CPU | train loss 0.0015, val accuracy 0.998 |
| Spiral MLP, SGD + momentum, lr 0.1, 60 epochs, CPU | train loss 0.0018, val accuracy 0.997 |
| `scratchnet.AdamW` vs `torch.optim.AdamW`, largest epoch-loss gap over 60 epochs | 0.0 |
| `scratchnet.SGD` vs `torch.optim.SGD`, largest epoch-loss gap over 60 epochs | 0.0 |
| `optimizer.step()`, 6.3 M parameters, CPU: ours / torch for-loop / foreach / fused | 11.7 / 8.3 / 10.0 / 4.4 ms |
| `optimizer.step()`, 6.3 M parameters, MPS: ours / torch for-loop / foreach / fused | 8.6 / 11.2 / 10.9 / 5.5 ms |
| 2048 x 2048 `torch.mm`: CPU, 2 threads / MPS | 35.0 / 6.0 ms |
| 4096 x 4096 `torch.mm` on MPS, without / with `synchronize()` | 0.285 / 53.6 ms |
| 256 MB copy to MPS: pageable / pinned; copy back | 15.2 / 21.7 GB/s; 14.0 GB/s |
| DataLoader, 2 ms per example, epoch 2: 0 workers / 2 persistent / 4 persistent | 495 / 945 / 1767 examples/s |

No GPU numbers are reported because this machine has no CUDA device. To measure them, run the same benchmarks on a
CUDA machine: `pick_device` selects CUDA first and `synchronize` calls `torch.cuda.synchronize`.

## Theory

- [What Autograd Actually Does](https://srini-anirudh.github.io/blogs/what-autograd-actually-does/)
- [What an Optimizer Actually Does](https://srini-anirudh.github.io/blogs/what-an-optimizer-actually-does/)
