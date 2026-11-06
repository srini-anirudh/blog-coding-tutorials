"""Three precision facts from the post, each made to happen.

1. A BF16 accumulator loses a long sum; BF16 products summed in FP32 do not.
2. A small update to a BF16 weight rounds to nothing; the same update to an FP32 master survives.
3. FP16 flushes small gradients to zero unless the loss is scaled; BF16 keeps FP32's range.
"""

from __future__ import annotations

import torch
from torch import nn

from .mixed import MasterWeightAdam


def dot_errors(n: int = 4096, seed: int = 0) -> dict[str, float]:
    """Relative error of a length-n dot product of BF16 inputs, under different accumulators."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g).bfloat16()
    y = torch.rand(n, generator=g).bfloat16()
    exact = (x.double() * y.double()).sum().item()

    acc16 = torch.zeros((), dtype=torch.bfloat16)
    acc32 = torch.zeros((), dtype=torch.float32)
    for a, b in zip(x, y):
        prod = a * b  # a BF16 multiply, rounded to BF16
        acc16 = acc16 + prod  # running sum rounded to BF16 after every add
        acc32 = acc32 + prod.float()  # what a tensor core does: BF16 products, FP32 sum
    lib = (x @ y).item()  # PyTorch's own BF16 dot product
    rel = lambda v: abs(v - exact) / exact  # noqa: E731
    return {"exact": exact, "bf16 accumulate": rel(acc16.item()), "fp32 accumulate": rel(acc32.item()),
            "torch bf16 matmul": rel(lib)}


def ones_sum_stall(n: int = 4096) -> float:
    """Add 1.0 n times in BF16. Past 256 the spacing is 2, so 256 + 1 rounds back to 256."""
    acc = torch.zeros((), dtype=torch.bfloat16)
    one = torch.ones((), dtype=torch.bfloat16)
    for _ in range(n):
        acc = acc + one
    return acc.item()


def update_survival(rel_update: float = 1e-4, steps: int = 1000) -> dict[str, float]:
    """Apply the same small relative update to a weight of 1.0 kept in BF16 and kept in FP32."""
    w16 = torch.ones((), dtype=torch.bfloat16)
    w32 = torch.ones((), dtype=torch.float32)
    for _ in range(steps):
        w16 = w16 - rel_update * w16
        w32 = w32 - rel_update * w32
    return {"bf16 weight": w16.item(), "fp32 master": w32.item(), "exact": (1 - rel_update) ** steps}


def regression_run(recipe: str, steps: int = 600, lr: float = 3e-4, seed: int = 0) -> list[float]:
    """Fit a linear layer whose weights sit near 1.0 to targets 1% away, with Adam.

    recipe 'pure_bf16': BF16 weights and BF16 Adam state, updated in place (torch.optim.AdamW).
    recipe 'master':    BF16 weights for compute, FP32 master and moments (MasterWeightAdam).
    recipe 'fp32':      everything FP32, the reference.
    """
    g = torch.Generator().manual_seed(seed)
    d = 64
    w_true = 1.0 + 0.01 * torch.randn(d, d, generator=g)
    x = torch.randn(512, d, generator=g)
    y = x @ w_true.T
    dtype = torch.float32 if recipe == "fp32" else torch.bfloat16
    lin = nn.Linear(d, d, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.ones(d, d))
    lin = lin.to(dtype)
    if recipe == "master":
        opt = MasterWeightAdam(lin.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        opt = torch.optim.AdamW(lin.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    losses = []
    for _ in range(steps):
        loss = ((lin(x.to(dtype)).float() - y) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses


def range_flush(seed: int = 0, loss_scale: float = 2.0**16) -> dict[str, float]:
    """Fraction of gradient-sized values (1e-10 to 1e-2, log-uniform) that become zero when cast."""
    g = torch.Generator().manual_seed(seed)
    vals = 10 ** (torch.rand(100_000, generator=g, dtype=torch.float64) * -8 - 2)
    vals = vals.float()
    zero = lambda t: (t == 0).double().mean().item()  # noqa: E731
    return {
        "fp16": zero(vals.half()),
        "fp16, loss scaled": zero((vals * loss_scale).half()),
        "bf16": zero(vals.bfloat16()),
        "fp16 max": torch.finfo(torch.float16).max,
        "bf16 max": torch.finfo(torch.bfloat16).max,
    }
