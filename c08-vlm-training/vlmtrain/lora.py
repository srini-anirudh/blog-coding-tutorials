"""LoRA for a VLM: one wrapper, placed on whichever modules a regex selects.

    y = W x + (alpha / r) * B (A x)       A: (r, d_in) random,  B: (d_out, r) zeros

B starts at zero, so a freshly wrapped model computes exactly what it did before. Placement
is the interesting choice in a VLM: LoRA on the LM, on the vision tower, on the projector, or
on combinations. tests/test_lora.py checks this module against PEFT.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn

TARGETS = {
    "lm": r"^lm\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    "vision": r"^vision\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$",
    "projector": r"^projector\..*",
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # PEFT's default initialisation for A
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., d_in) -> (..., d_out). The low-rank path costs r * (d_in + d_out) per token."""
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Fold B A into W for inference: no extra latency, no extra modules."""
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        out.to(self.base.weight.device, self.base.weight.dtype)
        out.weight.copy_(self.base.weight + self.scale * self.lora_B @ self.lora_A)
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def apply_lora(model: nn.Module, placements: list[str], r: int = 8, alpha: float = 16.0,
               dropout: float = 0.0) -> list[str]:
    """Wrap every nn.Linear whose full name matches one of the placements. Returns wrapped names."""
    patterns = [re.compile(TARGETS[p]) for p in placements]
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and any(p.match(n) for p in patterns)]
    for n in names:
        parent_name, _, child = n.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))
    return names


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear with its merged nn.Linear. Returns how many were merged."""
    wrapped = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for n in wrapped:
        parent_name, _, child = n.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, getattr(parent, child).merged())
    return len(wrapped)
