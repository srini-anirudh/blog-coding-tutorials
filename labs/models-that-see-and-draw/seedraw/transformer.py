"""One tiny pre-norm transformer that accepts any boolean attention mask.

`allowed[i, j] = True` means position i may attend to position j. The token priors pass a causal mask
(raster autoregression) or no mask (MaskGIT). The hybrid model passes the Transfusion mask.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def causal_mask(n: int) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool).tril()


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, allowed: torch.Tensor | None) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.n2(x))


class Trunk(nn.Module):
    """Learned positions plus a stack of blocks. Input and output are (batch, n, d) vectors."""

    def __init__(self, n_pos: int, d: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(n_pos, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, allowed: torch.Tensor | None = None) -> torch.Tensor:
        h = h + self.pos[: h.shape[1]]
        for blk in self.blocks:
            h = blk(h, allowed)
        return self.norm(h)
