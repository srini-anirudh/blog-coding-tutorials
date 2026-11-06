"""A minimal LoRA, written out so we can see what PEFT does.

    y = W x + (alpha / r) * B (A x)        W frozen (out, in),  A (r, in),  B (out, r),  B starts at zero

tests/test_lora.py checks this layer against peft.LoraConfig with the same A and B.
"""

from __future__ import annotations

import math

import torch
from torch import nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # same init as PEFT
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in) -> (..., out). B = 0 at init, so the wrapped model starts exactly at the base model.
        return self.base(x) + self.scale * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold B A into W so inference costs nothing extra."""
        self.base.weight += self.scale * (self.lora_B @ self.lora_A).to(self.base.weight.dtype)
        return self.base


def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, targets: tuple[str, ...] = DEFAULT_TARGETS) -> nn.Module:
    """Freeze every weight, then wrap each target nn.Linear in a LoRALinear (in place)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha))
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter weights: this is the whole LoRA checkpoint."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def merge_lora(model: nn.Module) -> nn.Module:
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merge())
    return model
