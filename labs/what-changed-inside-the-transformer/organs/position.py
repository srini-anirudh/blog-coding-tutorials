"""Position: the RoPE relative-offset property, the base as a context dial, and NoPE under a causal mask."""

from __future__ import annotations

import math

import torch

from . import _paths  # noqa: F401
from tinyllm.rope import RotaryEmbedding, rotate


def rope_logit(q: torch.Tensor, k: torch.Tensor, m: int, n: int, base: float = 10_000.0) -> float:
    """<R_m q, R_n k> for single vectors q, k of even length d, using C4's rotate-half RoPE."""
    d = q.shape[-1]
    rope = RotaryEmbedding(d, max(m, n) + 1, base)
    cos, sin = rope(torch.tensor([[m, n]]))
    rq = rotate(q.view(1, 1, 1, d).expand(1, 1, 2, d), cos, sin)[0, 0, 0]
    rk = rotate(k.view(1, 1, 1, d).expand(1, 1, 2, d), cos, sin)[0, 0, 1]
    return (rq * rk).sum().item()


def rope_wavelengths(head_dim: int, base: float) -> torch.Tensor:
    """Tokens per full turn of each pair: 2 pi / theta_i with theta_i = base^(-2i/d)."""
    i = torch.arange(head_dim // 2, dtype=torch.float64)
    return 2 * math.pi * base ** (2 * i / head_dim)


def nope_attention(x: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, causal: bool) -> torch.Tensor:
    """One attention head with no position signal at all. x: (B, T, d) -> (B, T, d_v)."""
    q, k, v = x @ wq, x @ wk, x @ wv
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        t = x.shape[1]
        s = s.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf"))
    return s.softmax(-1) @ v


def position_probe_r2(feats: torch.Tensor, target: torch.Tensor, split: float = 0.5, ridge: float = 1e-3) -> float:
    """Held-out R^2 of a ridge regression from features (N, F) to a scalar target (N,)."""
    n = int(split * feats.shape[0])
    x = torch.cat([feats, torch.ones(feats.shape[0], 1, dtype=feats.dtype)], 1)
    a, b = x[:n], target[:n]
    w = torch.linalg.solve(a.T @ a + ridge * torch.eye(a.shape[1], dtype=a.dtype), a.T @ b)
    pred, y = x[n:] @ w, target[n:]
    return (1 - (y - pred).pow(2).sum() / (y - y.mean()).pow(2).sum()).item()
