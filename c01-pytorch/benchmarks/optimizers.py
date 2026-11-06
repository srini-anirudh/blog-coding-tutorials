"""optimizer.step() cost: our readable AdamW vs torch's for-loop, foreach and fused kernels.

Run: python c01-pytorch/benchmarks/optimizers.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from scratchnet.device import pick_device, synchronize
from scratchnet.optim import AdamW

torch.set_num_threads(2)


def make_params(device: torch.device, layers: int = 24, width: int = 512) -> list[nn.Parameter]:
    """layers x (W [width, width] + b [width]): many mid-sized tensors, like a real model."""
    params = []
    for _ in range(layers):
        for shape in ((width, width), (width,)):
            p = nn.Parameter(torch.randn(shape, device=device) * 0.02)
            p.grad = torch.randn_like(p) * 1e-3
            params.append(p)
    return params


def bench_step(make_opt, device, steps: int = 20) -> float:
    params = make_params(device)
    opt = make_opt(params)
    for _ in range(3):  # warm-up creates the state tensors
        opt.step()
    synchronize(device)
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        opt.step()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    devices = [torch.device("cpu")]
    if pick_device().type != "cpu":
        devices.append(pick_device())
    n_params = sum(p.numel() for p in make_params(torch.device("cpu")))
    print(f"{n_params / 1e6:.2f} M parameters in {len(make_params(torch.device('cpu')))} tensors")
    impls = {
        "scratchnet.AdamW": lambda ps: AdamW(ps, lr=1e-3),
        "torch for-loop": lambda ps: torch.optim.AdamW(ps, lr=1e-3, foreach=False),
        "torch foreach": lambda ps: torch.optim.AdamW(ps, lr=1e-3, foreach=True),
        "torch fused": lambda ps: torch.optim.AdamW(ps, lr=1e-3, fused=True),
    }
    print(f"{'impl':<18}" + "".join(f"{d.type + ' ms':>10}" for d in devices))
    for name, make in impls.items():
        cells = []
        for d in devices:
            try:
                cells.append(f"{bench_step(make, d) * 1e3:>10.2f}")
            except (RuntimeError, ValueError) as err:
                cells.append(f"{'n/a':>10}")
                print(f"  ({name} on {d.type}: {str(err).splitlines()[0][:70]})")
        print(f"{name:<18}" + "".join(cells))


if __name__ == "__main__":
    main()
