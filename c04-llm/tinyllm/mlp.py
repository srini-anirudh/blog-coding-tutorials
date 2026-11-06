"""SwiGLU feed-forward block: down( silu(x W_gate) * (x W_up) )."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """(B, T, D) -> (B, T, F) twice -> gate -> (B, T, D).

    Three matrices instead of the classic two. To keep the parameter count of a 4*D MLP,
    Llama-style models pick F close to (2/3) * 4 * D, rounded to a hardware-friendly multiple.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The gate decides, per channel and per token, how much of the "up" value passes.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
