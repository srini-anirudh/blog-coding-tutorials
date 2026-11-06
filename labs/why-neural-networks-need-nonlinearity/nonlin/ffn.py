"""Transformer feed-forward blocks: two-matmul elementwise FFNs and three-matmul gated FFNs (section 05).

Ungated:  down( act(x W_up) )                       2 * d * h parameters
Gated:    down( act(x W_gate) * (x W_up) )          3 * d * h parameters

`matched_hidden` picks h so that every variant has the same parameter count as a 4d ungated FFN. That is the
8d/3 rule. Each block remembers the tensor it fed into `down_proj` on its last call, because that tensor is where
the post's sparsity (section 06) and outlier (section 05) claims live.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .activations import get

# name -> (activation applied to the gate or to the single projection, gated?)
FFN_KINDS: dict[str, tuple[str, bool]] = {
    "relu": ("relu", False),
    "gelu": ("gelu", False),
    "relu2": ("relu2", False),
    "bilinear": ("identity", True),
    "reglu": ("relu", True),
    "geglu": ("gelu", True),
    "swiglu": ("silu", True),
}


def round_up(n: float, multiple: int) -> int:
    return int(math.ceil(n / multiple) * multiple)


def matched_hidden(d: int, kind: str, mult: float = 4.0, multiple_of: int = 1) -> int:
    """Hidden width with the same parameter budget as an ungated FFN of width mult * d."""
    gated = FFN_KINDS[kind][1]
    h = mult * d * (2.0 / 3.0 if gated else 1.0)
    return round_up(h, multiple_of)


class FFN(nn.Module):
    def __init__(self, d: int, h: int, kind: str = "swiglu", record: bool = False) -> None:
        super().__init__()
        if kind not in FFN_KINDS:
            raise ValueError(f"unknown FFN kind {kind!r}; choose from {sorted(FFN_KINDS)}")
        act_name, self.gated = FFN_KINDS[kind]
        self.kind = kind
        self.act = get(act_name)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.gate_proj = nn.Linear(d, h, bias=False) if self.gated else None
        self.down_proj = nn.Linear(h, d, bias=False)
        self.record = record
        self.last_hidden: torch.Tensor | None = None

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The (..., h) tensor that goes into down_proj."""
        if self.gated:
            return self.act(self.gate_proj(x)) * self.up_proj(x)
        return self.act(self.up_proj(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.hidden(x)
        if self.record:
            self.last_hidden = a.detach()
        return self.down_proj(a)


def ffn_params(d: int, h: int, gated: bool) -> int:
    return (3 if gated else 2) * d * h


def sparse_down(a: torch.Tensor, w_down_t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """down_proj for one token that reads only the rows of W_down^T whose activation is nonzero.

    a: (h,) activation.   w_down_t: (h, d), the transpose of the nn.Linear weight, so each skipped unit is one
    contiguous row that never leaves memory. Returns the (d,) output and how many rows were read.
    Exact zeros contribute nothing, so the result equals the dense product. That is why exact zeros (ReLU) let a
    kernel skip work and near-zeros (GELU, SiLU) do not.
    """
    idx = torch.nonzero(a, as_tuple=True)[0]
    return a.index_select(0, idx) @ w_down_t.index_select(0, idx), int(idx.numel())
