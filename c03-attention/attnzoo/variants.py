"""Research variants, each a short change to standard attention.

They are not equally mainstream:
    GatedAttention        output gate per head; research (Qiu et al., 2025)
    CosineAttention       normalised q and k with a learned temperature; Swin V2, nGPT-like
    DifferentialAttention difference of two softmax maps; research (Ye et al., 2024)
    TalkingHeadsAttention mixes heads before and after softmax; research (Shazeer et al., 2020)
All take hidden (B, N, D) -> (B, N, D) and use plain causal or bidirectional masks, no RoPE.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .backends import attend
from .config import AttentionConfig
from .heads import merge_heads, split_heads
from .masks import causal_mask
from .stability import RMSNorm


class _QKV(nn.Module):
    """Shared projections for the variants: q, k, v (B, H, N, dh) and an output projection."""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        d, h, dh = config.hidden_size, config.num_heads, config.head_dim
        self.config, self.h = config, h
        self.qkv = nn.Linear(d, 3 * h * dh, bias=False)
        self.o_proj = nn.Linear(h * dh, d, bias=False)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(split_heads(t, self.h) for t in self.qkv(x).chunk(3, dim=-1))

    def mask(self, n: int, device: torch.device) -> torch.Tensor | None:
        return causal_mask(n, device=device) if self.config.causal else None


class GatedAttention(_QKV):
    """out_h <- out_h * sigmoid(x W_g)_h, elementwise, before the output projection.

    The gate lets a head output nothing, which removes the need for "attention sinks"
    (heads parking weight on token 0 to do nothing).
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__(config)
        self.gate = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project(x)
        out = merge_heads(attend(q, k, v, backend=self.config.backend, causal=self.config.causal))
        return self.o_proj(out * torch.sigmoid(self.gate(x)))


class CosineAttention(_QKV):
    """logits = tau * cos(q, k). Logits are bounded by tau, whatever the activation scale."""

    def __init__(self, config: AttentionConfig, init_tau: float = 10.0) -> None:
        super().__init__(config)
        self.log_tau = nn.Parameter(torch.full((config.num_heads, 1, 1), math.log(init_tau)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project(x)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        # tau differs per head, so fold it into q instead of passing a scalar scale.
        out = attend(q * self.log_tau.exp(), k, v, backend=self.config.backend, causal=self.config.causal, scale=1.0)
        return self.o_proj(merge_heads(out))


class DifferentialAttention(_QKV):
    """A = softmax(Q1 K1^T) - lambda * softmax(Q2 K2^T); out = A V (simplified Diff Transformer).

    Each head splits its q and k into two halves. Noise that both maps share cancels.
    """

    def __init__(self, config: AttentionConfig, lambda_init: float = 0.8) -> None:
        super().__init__(config)
        self.lambda_init = lambda_init
        self.lam = nn.Parameter(torch.tensor(lambda_init))
        self.head_norm = RMSNorm(config.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project(x)
        (q1, q2), (k1, k2) = q.chunk(2, dim=-1), k.chunk(2, dim=-1)
        scale = 1.0 / math.sqrt(q1.size(-1))
        mask = self.mask(x.size(1), x.device)
        maps = []
        for qi, ki in ((q1, k1), (q2, k2)):
            s = (qi @ ki.transpose(-2, -1)) * scale
            if mask is not None:
                s = s.masked_fill(~mask, float("-inf"))
            maps.append(torch.softmax(s, dim=-1))
        a = maps[0] - self.lam * maps[1]  # can be negative: it is no longer a distribution
        out = self.head_norm(a @ v) * (1.0 - self.lambda_init)
        return self.o_proj(merge_heads(out))


class TalkingHeadsAttention(_QKV):
    """Mix the H logit maps with a learned H x H matrix before softmax, and the H weight maps after."""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__(config)
        h = config.num_heads
        self.pre = nn.Parameter(torch.eye(h))  # identity at init: starts as plain MHA
        self.post = nn.Parameter(torch.eye(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project(x)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # (B, H, N, N)
        s = torch.einsum("bhnm,gh->bgnm", s, self.pre)
        mask = self.mask(x.size(1), x.device)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        w = torch.einsum("bhnm,gh->bgnm", torch.softmax(s, dim=-1), self.post)
        return self.o_proj(merge_heads(w @ v))
