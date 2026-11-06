"""RMSNorm: rescale each token vector to unit root-mean-square, then apply a learned gain."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight, computed per token.

    Input and output: (..., hidden_size). Unlike LayerNorm there is no mean subtraction and no bias.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Square-and-mean in float32: in bf16 the sum of squares loses too many bits.
        x32 = x.float()
        rms_inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (..., 1)
        return self.weight * (x32 * rms_inv).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"
