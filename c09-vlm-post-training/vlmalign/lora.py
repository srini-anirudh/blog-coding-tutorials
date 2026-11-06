"""A minimal LoRA, so the trainable set is explicit and the reference model is free.

W x  ->  W x + (alpha / r) * B A x,  with A: (r, d_in) random, B: (d_out, r) zero.
B = 0 at the start, so the adapted model equals the pretrained one exactly.
Turning adapters off recovers the starting model, which is the DPO/GRPO
reference without a second copy of the weights. (C6 derives LoRA; here we use it.)
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # same init as PEFT
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return out


def add_lora(model: nn.Module, pattern: str, r: int = 16, alpha: int = 32) -> list[str]:
    """Wrap every nn.Linear whose qualified name matches `pattern`. Returns wrapped names."""
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and re.search(pattern, full):
                setattr(module, child_name, LoRALinear(child, r, alpha))
                wrapped.append(full)
    return wrapped


@contextmanager
def lora_disabled(model: nn.Module):
    """Temporarily run the base weights only (the reference policy)."""
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    for m in layers:
        m.enabled = False
    try:
        yield
    finally:
        for m in layers:
            m.enabled = True


def lora_delta_norms(model: nn.Module, prefix: str) -> float:
    """Total Frobenius norm of scale * B A over adapted layers under `prefix`: how far training moved them."""
    total = 0.0
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear) and name.startswith(prefix):
            total += float((m.scale * m.lora_B @ m.lora_A).float().pow(2).sum())
    return math.sqrt(total)
