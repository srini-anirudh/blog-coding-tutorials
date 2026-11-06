"""Where 1/sqrt(d_k) comes from: logit variance, softmax entropy and the gradient through softmax.

With q and k drawn with independent unit-variance entries, Var[q . k] = d_k. Unscaled logits spread
by sqrt(d_k), softmax saturates towards one-hot, and its Jacobian diag(a) - a a^T goes to zero.
"""

from __future__ import annotations

import math

import torch


def softmax_stats(d_k: int, n_keys: int = 128, n_queries: int = 2000, scaled: bool = True, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n_queries, d_k, generator=g)
    k = torch.randn(n_keys, d_k, generator=g)
    logits = q @ k.T
    if scaled:
        logits = logits / math.sqrt(d_k)
    a = torch.softmax(logits, dim=-1)
    entropy = -(a * a.clamp_min(1e-30).log()).sum(-1)
    # Frobenius norm of the softmax Jacobian, diag(a) - a a^T, for each query row.
    jac = (a.pow(2).sum(-1) - 2 * a.pow(3).sum(-1) + a.pow(2).sum(-1).pow(2)).clamp_min(0).sqrt()
    return {
        "d_k": d_k,
        "logit std": logits.std().item(),
        "entropy (nats)": entropy.mean().item(),
        "max weight": a.max(-1).values.mean().item(),
        "rows with max weight > 0.99": (a.max(-1).values > 0.99).float().mean().item(),
        "|J_softmax|_F": jac.mean().item(),
    }


def jacobian_norm_direct(a: torch.Tensor) -> torch.Tensor:
    """The same Frobenius norm, built from the full matrix, for testing the closed form above."""
    jac = torch.diag_embed(a) - a[..., :, None] * a[..., None, :]
    return jac.flatten(-2).norm(dim=-1)


def grad_to_queries(d_k: int, scaled: bool, n_keys: int = 128, seed: int = 0) -> float:
    """Mean |dL/dq| for a random upstream gradient on the attention output: what training sees."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(512, d_k, generator=g, requires_grad=True)
    k = torch.randn(n_keys, d_k, generator=g)
    v = torch.randn(n_keys, d_k, generator=g)
    up = torch.randn(512, d_k, generator=g)
    logits = q @ k.T / (math.sqrt(d_k) if scaled else 1.0)
    (torch.softmax(logits, -1) @ v * up).sum().backward()
    return q.grad.norm(dim=-1).mean().item()


def shift_invariance(seed: int = 0) -> float:
    """softmax(s + c) = softmax(s): only score differences matter. The online softmax relies on it."""
    s = torch.randn(4, 16, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return (torch.softmax(s, -1) - torch.softmax(s + 123.0, -1)).abs().max().item()
