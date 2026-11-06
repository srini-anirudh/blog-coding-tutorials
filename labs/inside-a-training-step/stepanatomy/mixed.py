"""The post's mixed-precision recipe, written out: BF16 compute copy, FP32 master weights, FP32 Adam.

Per parameter: 2 bytes of BF16 weight, 2 of BF16 gradient, 4 of FP32 master, 4 + 4 of Adam moments.
The optimizer reads the BF16 gradient, updates the FP32 master, and casts it back down.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MasterWeightAdam:
    """AdamW whose state lives in FP32 next to a low-precision model.

    The arithmetic follows torch.optim.AdamW line for line, so the two agree when given the same
    FP32 gradients. The FP32 copy of each gradient is a per-tensor temporary (workspace), not state.
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.master = [p.detach().to(torch.float32, copy=True) for p in self.params]
        self.m = [torch.zeros_like(w) for w in self.master]
        self.v = [torch.zeros_like(w) for w in self.master]
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.t = 0

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1**self.t, 1 - b2**self.t
        for p, w, m, v in zip(self.params, self.master, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad.float()
            w.mul_(1 - self.lr * self.wd)
            m.lerp_(g, 1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (v.sqrt() / math.sqrt(bc2)).add_(self.eps)
            w.addcdiv_(m, denom, value=-self.lr / bc1)
            p.copy_(w)  # the cast down to the compute dtype

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def ledger(model: nn.Module, opt: MasterWeightAdam) -> dict[str, float]:
    """Bytes per parameter in each category, counted from the tensors themselves."""
    n = sum(p.numel() for p in opt.params)

    def total(ts):
        return sum(t.numel() * t.element_size() for t in ts if t is not None)

    out = {
        f"params ({opt.params[0].dtype})": total(opt.params) / n,
        "grads": total(p.grad for p in opt.params) / n,
        "master (fp32)": total(opt.master) / n,
        "adam m (fp32)": total(opt.m) / n,
        "adam v (fp32)": total(opt.v) / n,
    }
    out["total"] = sum(out.values())
    return out
