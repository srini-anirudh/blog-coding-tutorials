"""Four answers to "who attends to whom", each in one small module.

MHSA                 ViT's global multi-head self-attention, N x N scores
SRAttention          PVT: queries at full resolution, keys and values downsampled by R, N x N/R^2 scores
PoolingAttention     MViT: pool Q, K and V separately; stride_q downsamples, stride_kv shrinks the key budget
GPSA                 ConViT: a per-head gate mixes content attention with a purely positional one

Every module takes tokens (B, N, D). The spatial ones also take the grid side (h, w).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def split(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, n, d = x.shape
    return x.view(b, n, heads, d // heads).transpose(1, 2)  # (B, h, N, d)


def merge(x: torch.Tensor) -> torch.Tensor:
    b, h, n, d = x.shape
    return x.transpose(1, 2).reshape(b, n, h * d)


def attention_weights(q: torch.Tensor, k: torch.Tensor, bias: torch.Tensor | None = None, scale: bool = True):
    """softmax(q k^T / sqrt(d) + bias). Materialises the (B, h, Nq, Nk) matrix on purpose."""
    s = q @ k.transpose(-2, -1)
    if scale:
        s = s / math.sqrt(q.size(-1))
    if bias is not None:
        s = s + bias
    return s.softmax(dim=-1)


class MHSA(nn.Module):
    """Global self-attention. return_attn=True takes the explicit path and also returns the weights."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        q, k, v = (split(t, self.heads) for t in self.qkv(x).chunk(3, dim=-1))
        if return_attn:
            a = attention_weights(q, k)
            return self.proj(merge(a @ v)), a
        return self.proj(merge(F.scaled_dot_product_attention(q, k, v)))


class SRAttention(nn.Module):
    """PVT v1 spatial-reduction attention. A strided R x R conv turns h*w key tokens into h*w / R^2."""

    def __init__(self, dim: int, heads: int, ratio: int) -> None:
        super().__init__()
        self.heads, self.ratio = heads, ratio
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        if ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=ratio, stride=ratio)
            self.norm = nn.LayerNorm(dim)

    def reduce(self, x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        if self.ratio == 1:
            return x
        b, n, d = x.shape
        x = self.sr(x.transpose(1, 2).reshape(b, d, *hw)).flatten(2).transpose(1, 2)
        return self.norm(x)

    def forward(self, x: torch.Tensor, hw: tuple[int, int], return_attn: bool = False):
        q = split(self.q(x), self.heads)
        k, v = (split(t, self.heads) for t in self.kv(self.reduce(x, hw)).chunk(2, dim=-1))
        if return_attn:
            a = attention_weights(q, k)  # (B, h, N, N / R^2): not square
            return self.proj(merge(a @ v)), a
        return self.proj(merge(F.scaled_dot_product_attention(q, k, v)))


class Pool(nn.Module):
    """MViT's pooling operator: a depthwise conv with stride s, then LayerNorm. s = 1 is the identity."""

    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        if stride > 1:
            k = 2 * (stride // 2) + 1
            self.conv = nn.Conv2d(dim, dim, k, stride=stride, padding=stride // 2, groups=dim, bias=False)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, hw: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.stride == 1:
            return x, hw
        b, n, d = x.shape
        y = self.conv(x.transpose(1, 2).reshape(b, d, *hw))
        new_hw = (y.size(2), y.size(3))
        return self.norm(y.flatten(2).transpose(1, 2)), new_hw


class PoolingAttention(nn.Module):
    """MViT(v2) pooling attention. Output has the query's resolution, so stride_q = 2 downsamples."""

    def __init__(self, dim: int, heads: int, stride_q: int = 1, stride_kv: int = 1, residual_pool: bool = True) -> None:
        super().__init__()
        self.heads, self.residual_pool = heads, residual_pool
        self.qkv = nn.Linear(dim, 3 * dim)
        self.pool_q, self.pool_k, self.pool_v = Pool(dim, stride_q), Pool(dim, stride_kv), Pool(dim, stride_kv)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, hw: tuple[int, int]):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, out_hw = self.pool_q(q, hw)
        k, _ = self.pool_k(k, hw)
        v, _ = self.pool_v(v, hw)
        q, k, v = split(q, self.heads), split(k, self.heads), split(v, self.heads)
        out = F.scaled_dot_product_attention(q, k, v)
        if self.residual_pool:  # MViTv2: add the pooled query back
            out = out + q
        return self.proj(merge(out)), out_hw


def relative_offsets(side: int) -> torch.Tensor:
    """(N, N, 3) with r_ij = (dx, dy, dx^2 + dy^2) between patches i and j of a side x side grid."""
    ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    ys, xs = ys.flatten().float(), xs.flatten().float()
    dx, dy = xs[None, :] - xs[:, None], ys[None, :] - ys[:, None]
    return torch.stack([dx, dy, dx**2 + dy**2], dim=-1)


class GPSA(nn.Module):
    """ConViT gated positional self-attention, patch tokens only (no CLS).

    A = (1 - sigmoid(lambda_h)) * softmax(q k^T / sqrt(d)) + sigmoid(lambda_h) * softmax(v_pos . r_ij)
    With local_init, head h's positional map is a Gaussian-like bump at its own offset Delta_h,
    so at step zero the layer behaves like a convolution with a sqrt(h) x sqrt(h) kernel.
    """

    def __init__(self, dim: int, heads: int, side: int, gate_init: float = 1.0, locality_strength: float = 1.0) -> None:
        super().__init__()
        self.heads = heads
        self.qk = nn.Linear(dim, 2 * dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, heads)
        self.gate = nn.Parameter(torch.full((heads,), gate_init))
        self.register_buffer("rel", relative_offsets(side), persistent=False)
        self.local_init(locality_strength)

    @torch.no_grad()
    def local_init(self, strength: float) -> None:
        """Score_h(i, j) = -strength * |delta_ij - Delta_h|^2 + const, Delta_h on a sqrt(h) grid."""
        k = int(self.heads**0.5)
        center = (k - 1) / 2
        self.pos_proj.weight.zero_()
        self.pos_proj.bias.zero_()
        for h1 in range(k):
            for h2 in range(k):
                h = h1 + k * h2
                self.pos_proj.weight[h] = torch.tensor([2 * (h1 - center), 2 * (h2 - center), -1.0]) * strength

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        q, k = (split(t, self.heads) for t in self.qk(x).chunk(2, dim=-1))
        v = split(self.v(x), self.heads)
        content = attention_weights(q, k)  # (B, h, N, N)
        positional = self.pos_proj(self.rel).permute(2, 0, 1).softmax(dim=-1)[None]  # (1, h, N, N)
        g = torch.sigmoid(self.gate)[None, :, None, None]
        a = (1 - g) * content + g * positional
        a = a / a.sum(dim=-1, keepdim=True)
        out = self.proj(merge(a @ v))
        return (out, a) if return_attn else out
