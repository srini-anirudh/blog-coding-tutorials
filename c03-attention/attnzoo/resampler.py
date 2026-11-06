"""Learned-query bottlenecks: turn Nv image tokens into M << Nv tokens for the LM.

QFormer: learned queries self-attend, then cross-attend to the image (BLIP-2).
PerceiverResampler: learned latents cross-attend to [image; latents] (Flamingo).
Both output (B, M, D) for any Nv, so the language model's cost no longer depends
on image resolution.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import AttentionConfig
from .cross import CrossAttention, FeedForward
from .gqa import GroupedQueryAttention


class QFormerLayer(nn.Module):
    """queries <- queries + SelfAttn;  queries <- queries + CrossAttn(image);  queries <- queries + FFN."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        cfg = AttentionConfig("mha", hidden_size, num_heads, causal=False, positional_encoding="none")
        self.norm1, self.norm2 = nn.LayerNorm(hidden_size), nn.LayerNorm(hidden_size)
        self.self_attn = GroupedQueryAttention(cfg)  # queries talk to each other
        self.cross_attn = CrossAttention(cfg)  # queries read the image
        self.ffn = FeedForward(hidden_size)

    def forward(self, queries: torch.Tensor, image: torch.Tensor, image_mask: torch.Tensor | None) -> torch.Tensor:
        queries = queries + self.self_attn(self.norm1(queries))
        queries = queries + self.cross_attn(self.norm2(queries), image, image_mask)
        return queries + self.ffn(queries)


class QFormer(nn.Module):
    """image_tokens (B, Nv, Dv) -> (B, M, D) with M = num_queries learned query vectors."""

    def __init__(self, num_queries: int, hidden_size: int, vision_size: int, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)  # Q_L in R^{M x d}
        self.in_proj = nn.Linear(vision_size, hidden_size)
        self.layers = nn.ModuleList(QFormerLayer(hidden_size, num_heads) for _ in range(num_layers))

    def forward(self, image_tokens: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        image = self.in_proj(image_tokens)  # (B, Nv, D)
        q = self.queries.expand(image.size(0), -1, -1)  # (B, M, D): the same queries for every image
        for layer in self.layers:
            q = layer(q, image, image_mask)
        return q


class PerceiverResamplerLayer(nn.Module):
    """Latents query the concatenation [image; latents], so they also see each other."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        cfg = AttentionConfig("mha", hidden_size, num_heads, causal=False, positional_encoding="none")
        self.norm_media, self.norm_latents = nn.LayerNorm(hidden_size), nn.LayerNorm(hidden_size)
        self.cross_attn = CrossAttention(cfg)
        self.ffn = FeedForward(hidden_size)

    def forward(self, latents: torch.Tensor, image: torch.Tensor, image_mask: torch.Tensor | None) -> torch.Tensor:
        lat = self.norm_latents(latents)
        kv = torch.cat([self.norm_media(image), lat], dim=1)  # (B, Nv + M, D)
        mask = None
        if image_mask is not None:  # latents are never padding
            mask = torch.cat([image_mask, image_mask.new_ones(latents.shape[:2])], dim=1)
        latents = latents + self.cross_attn(lat, kv, mask)
        return latents + self.ffn(latents)


class PerceiverResampler(nn.Module):
    """image_tokens (B, Nv, Dv) -> (B, M, D) with M = num_latents."""

    def __init__(self, num_latents: int, hidden_size: int, vision_size: int, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_size) * 0.02)
        self.in_proj = nn.Linear(vision_size, hidden_size)
        self.layers = nn.ModuleList(PerceiverResamplerLayer(hidden_size, num_heads) for _ in range(num_layers))
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, image_tokens: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        image = self.in_proj(image_tokens)
        lat = self.latents.expand(image.size(0), -1, -1)
        for layer in self.layers:
            lat = layer(lat, image, image_mask)
        return self.norm(lat)
