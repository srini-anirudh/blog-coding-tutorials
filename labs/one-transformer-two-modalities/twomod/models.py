"""Two models around one middle. Only the first few lines and the last few lines of each forward differ.

TextTransformer   ids -> E[ids] + pos -> blocks (causal) -> norm -> tied head: logits at every position
ViT               image -> patches @ W + pos, [CLS] and registers -> blocks (no mask) -> norm -> one vector -> head
SmallCNN          the baseline with locality and translation equivariance built in
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .block import Block, BlockConfig, make_norm
from .front import PatchEmbed
from .positions import sincos_2d


class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        cfg: BlockConfig,
        depth: int,
        max_len: int = 512,
        pos: str = "learned",  # "learned" (GPT-2) or "none" (RoPE models add nothing to the stream)
        causal: bool = True,
        tie: bool = True,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.embed = nn.Embedding(vocab_size, cfg.dim)
        self.pos = nn.Parameter(torch.randn(max_len, cfg.dim) * 0.02) if pos == "learned" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(depth))
        self.norm = make_norm(cfg) if cfg.prenorm else nn.Identity()
        self.head = nn.Linear(cfg.dim, vocab_size, bias=False)
        if tie:
            self.head.weight = self.embed.weight  # the same parameters read and write
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, ids: torch.Tensor, caches: list[dict] | None = None, start: int = 0) -> torch.Tensor:
        """ids (B, N) -> logits (B, N, V). With `caches`, ids are the next tokens after `start` cached ones."""
        n = ids.shape[1]
        x = self.embed(ids)
        if self.pos is not None:
            x = x + self.pos[start : start + n]
        # The one boolean: a lower-triangular mask (offset by the cached prefix), or no mask at all.
        mask = torch.ones(n, start + n, dtype=torch.bool, device=ids.device).tril(start) if self.causal else None
        for i, block in enumerate(self.blocks):
            x = block(x, mask, caches[i] if caches is not None else None)
        return self.head(self.norm(x))

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        """Next-token cross-entropy: N - 1 supervised predictions from one sequence of N tokens."""
        logits = self(ids)
        return F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten())


class ViT(nn.Module):
    def __init__(
        self,
        cfg: BlockConfig,
        depth: int,
        image_size: int = 32,
        patch: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        pos: str = "learned",  # "learned" 1D table over raster order, "sincos2d" (fixed), or "none"
        pool: str = "cls",  # "cls" or "mean"
        registers: int = 0,
    ) -> None:
        super().__init__()
        self.grid = image_size // patch
        n = self.grid * self.grid
        self.pool, self.n_special = pool, (1 if pool == "cls" else 0) + registers
        self.patch_embed = PatchEmbed(patch, in_channels, cfg.dim)
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02) if pool == "cls" else None
        self.registers = nn.Parameter(torch.randn(1, registers, cfg.dim) * 0.02) if registers else None
        if pos == "learned":
            self.pos = nn.Parameter(torch.randn(n, cfg.dim) * 0.02)
        else:
            self.register_buffer("pos", sincos_2d(self.grid, self.grid, cfg.dim) if pos == "sincos2d" else None)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(depth))
        self.norm = make_norm(cfg) if cfg.prenorm else nn.Identity()
        self.head = nn.Linear(cfg.dim, num_classes)

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, C, H, W) -> the sequence the blocks see: (B, n_special + N, d)."""
        x = self.patch_embed(images)
        if self.pos is not None:
            x = x + self.pos  # position goes on patches; [CLS] and registers have no position
        special = [t.expand(x.shape[0], -1, -1) for t in (self.cls, self.registers) if t is not None]
        return torch.cat(special + [x], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.tokens(images)
        for block in self.blocks:
            x = block(x, None)  # no mask: every patch sees every patch from layer one
        x = self.norm(x)
        pooled = x[:, 0] if self.pool == "cls" else x[:, self.n_special :].mean(dim=1)
        return self.head(pooled)  # one prediction per image


class SmallCNN(nn.Module):
    """Three stages of two 3x3 convolutions. Locality and weight sharing across positions are hard-coded."""

    def __init__(self, num_classes: int = 10, widths: tuple[int, ...] = (64, 128, 256)) -> None:
        super().__init__()
        layers, c = [], 3
        for i, w in enumerate(widths):
            for _ in range(2):
                layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
                c = w
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images).mean(dim=(2, 3)))
