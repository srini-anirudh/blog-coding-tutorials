"""Tiny video models for the three training experiments.

BagOfFrames: encode each frame, average. It cannot know order: a permutation of the frames changes nothing.
TinyVideoTransformer: one token per frame, a learned query token at the end, and a choice of how time is
  marked: "none", "index" (frame number), "abstime" (seconds), or "text" (a timestamp token before each frame).
  A "text" timestamp such as 12.5 s is one token whose embedding is the sum of three digit embeddings
  (tens, units, tenths). This is a cheap stand-in for writing "<12.5 seconds>" as several text tokens: like
  real text, new values reuse digits the model has seen.
SpaceTimeCNN: a small convolution over the (time, x) slice of a video, for the playback-speed experiment.
Time enters the transformer as an additive sinusoid of the chosen number. The experiments compare which
number goes in, not RoPE against additive codes.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoid(pos: torch.Tensor, dim: int, max_period: float = 200.0) -> torch.Tensor:
    """Float positions (..., N) -> (..., N, dim) sin/cos code."""
    half = dim // 2
    freq = torch.exp(-math.log(max_period) * torch.arange(half, device=pos.device) / half)
    ang = pos[..., None].float() * freq
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class BagOfFrames(nn.Module):
    def __init__(self, in_dim: int, heads: dict[str, int], dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.heads = nn.ModuleDict({k: nn.Linear(dim, n) for k, n in heads.items()})

    def forward(self, frames: torch.Tensor, times=None, mask=None) -> dict[str, torch.Tensor]:
        h = self.enc(frames.flatten(2))
        if mask is None:
            pooled = h.mean(1)
        else:
            m = mask[..., None].float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return {k: head(pooled) for k, head in self.heads.items()}


class TinyVideoTransformer(nn.Module):
    def __init__(self, in_dim: int, heads: dict[str, int], time_mode: str = "index", dim: int = 64,
                 layers: int = 2, n_heads: int = 4, ticks_per_second: float = 2.0):
        super().__init__()
        if time_mode not in ("none", "index", "abstime", "text"):
            raise ValueError(time_mode)
        self.time_mode, self.dim, self.tps = time_mode, dim, ticks_per_second
        self.enc = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.digits = nn.ModuleList([nn.Embedding(10, dim) for _ in range(3)])  # tens, units, tenths
        block = nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleDict({k: nn.Linear(dim, n) for k, n in heads.items()})

    def stamp(self, times: torch.Tensor) -> torch.Tensor:
        """Seconds -> embedding of the written number: tens digit + units digit + tenths digit."""
        tenths = (times * 10).round().long().clamp(0, 999)
        parts = (tenths // 100, (tenths // 10) % 10, tenths % 10)
        return sum(emb(p) for emb, p in zip(self.digits, parts))

    def forward(self, frames: torch.Tensor, times: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        B, T = frames.shape[:2]
        x = self.enc(frames.flatten(2))
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=frames.device)
        if self.time_mode == "text":
            x = torch.stack([self.stamp(times), x], dim=2).reshape(B, 2 * T, self.dim)  # stamp, frame, ...
            mask = mask.repeat_interleave(2, dim=1)
        x = torch.cat([x, self.query.expand(B, 1, -1)], dim=1)
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.bool, device=mask.device)], dim=1)
        n = x.shape[1]
        if self.time_mode in ("index", "text"):
            pos = torch.arange(n, device=x.device).expand(B, -1)
            x = x + sinusoid(pos, self.dim)
        elif self.time_mode == "abstime":
            pos = torch.cat([times * self.tps, times.max(1, keepdim=True).values * self.tps + 1], dim=1)
            x = x + sinusoid(pos, self.dim)
        h = self.norm(self.body(x, src_key_padding_mask=~mask))[:, -1]
        return {k: head(h) for k, head in self.heads.items()}


class SpaceTimeCNN(nn.Module):
    """Reads speed as the slope of the blob's track in the (time, x) image."""

    def __init__(self, n_classes: int, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, padding_mode="circular"), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, n_classes))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (B, T, h, w) -> logits. The track is the height-averaged row of every frame."""
        return self.net(frames.mean(2)[:, None])
