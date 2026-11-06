"""A tiny VLM: patch encoder -> MLP connector -> causal decoder, plus LoRA for the decoder.

Each 8 x 8 cell of the 24 x 24 image is one patch, so an image is 9 visual tokens. They replace the
9 <image> placeholders in the token sequence just before the decoder runs, as in LLaVA.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chat import IMAGE
from .scenes import CELL, GRID

N_VISUAL = GRID * GRID


class Block(nn.Module):
    """Pre-norm transformer block. Separate q, k, v, o projections so LoRA can wrap each one."""

    def __init__(self, d: int, heads: int, causal: bool) -> None:
        super().__init__()
        self.heads, self.causal = heads, causal
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n1(x)
        q, k, v = (p(h).view(b, t, self.heads, d // self.heads).transpose(1, 2) for p in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.gelu(self.up(self.n2(x))))


class VisionEncoder(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.patch = nn.Linear(3 * CELL * CELL, d)
        self.pos = nn.Parameter(torch.randn(N_VISUAL, d) * 0.02)
        self.block = Block(d, 4, causal=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]  # (B, 3, 24, 24) -> (B, 9, 192) patches in reading order
        p = images.unfold(2, CELL, CELL).unfold(3, CELL, CELL).permute(0, 2, 3, 1, 4, 5).reshape(b, N_VISUAL, -1)
        return self.norm(self.block(self.patch(p) + self.pos))


class TinyVLM(nn.Module):
    def __init__(self, vocab_size: int, d: int = 64, layers: int = 3, heads: int = 4, max_len: int = 64) -> None:
        super().__init__()
        self.encoder = VisionEncoder(d)
        self.connector = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads, causal=True) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)

    def decoder_modules(self) -> list[nn.Module]:
        """The 'language model': everything except the encoder and the connector."""
        return [self.embed, self.pos, self.blocks, self.norm, self.head]

    def forward(self, ids: torch.Tensor, images: torch.Tensor | None = None) -> torch.Tensor:
        b, t = ids.shape
        x = self.embed(ids)
        is_img = ids == IMAGE
        if images is not None and is_img.any():
            vis = self.connector(self.encoder(images))  # (B, 9, d)
            slot = (is_img.cumsum(1) - 1).clamp(min=0, max=N_VISUAL - 1)  # k-th placeholder -> k-th token
            x = torch.where(is_img[..., None], vis.gather(1, slot[..., None].expand(-1, -1, x.shape[-1])), x)
        x = x + self.pos(torch.arange(t, device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen and B initialised to zero."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.a.T) @ self.b.T


def add_lora(model: TinyVLM, r: int = 8) -> None:
    """Freeze the decoder and wrap every linear layer inside its blocks with LoRA."""
    for m in model.decoder_modules():
        for p in m.parameters():
            p.requires_grad_(False)
    for blk in model.blocks:
        for name in ("q", "k", "v", "o", "up", "down"):
            setattr(blk, name, LoRALinear(getattr(blk, name), r))
