"""A residual MLP small enough to be launch bound or large enough to be GEMM bound, and a training step for it."""

from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm, self.up, self.down = nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


class ResMLP(nn.Module):
    def __init__(self, d: int, depth: int, n_classes: int = 16):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(d) for _ in range(depth)])
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.head(self.blocks(x))


def flops_per_step(d: int, depth: int, batch: int, n_classes: int = 16) -> float:
    """Useful training FLOPs from the model definition: 6 per weight per example (forward 2, backward 4)."""
    weights = depth * (2 * 4 * d * d) + d * n_classes
    return 6.0 * weights * batch


def make_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, dtype=None, sync: bool = True,
              fused: bool | None = None):
    """One optimizer step. `sync=True` reads the loss back to the host every step, the way a logging line does."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False if fused is None else None,
                            fused=fused)
    loss_fn = nn.CrossEntropyLoss()

    def step():
        with torch.autocast(x.device.type, dtype=dtype, enabled=dtype is not None):
            loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if sync:
            return loss.item()
        return loss

    return step
