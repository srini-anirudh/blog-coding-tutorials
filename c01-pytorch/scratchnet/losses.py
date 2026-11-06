"""Losses written from their definitions, checked against torch.nn.functional."""

from __future__ import annotations

import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """mean((pred - target)^2) over every element. Shapes must match exactly."""
    if pred.shape != target.shape:
        # F.mse_loss only warns here; a silent broadcast of [B, 1] against [B]
        # gives a [B, B] difference and a wrong loss that still trains.
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ")
    return ((pred - target) ** 2).mean()


def log_softmax_naive(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log(exp(z) / sum exp(z)). Correct on paper, overflows for logits above ~88 in float32."""
    e = logits.exp()
    return (e / e.sum(dim, keepdim=True)).log()


def log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log_softmax(z) = (z - m) - log(sum exp(z - m)), with m = max(z).

    Subtracting the max changes nothing mathematically (it cancels between
    numerator and denominator) but keeps every exp() argument <= 0.
    """
    m = logits.amax(dim, keepdim=True).detach()  # a constant shift; no gradient needed
    shifted = logits - m
    return shifted - shifted.exp().sum(dim, keepdim=True).log()


def cross_entropy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Mean negative log-likelihood of the target class.

    Shapes: logits [N, C] or [B, T, C]; target [N] or [B, T] of class ids.
    Positions whose target equals `ignore_index` contribute nothing and are
    not counted in the mean, like F.cross_entropy.
    """
    num_classes = logits.shape[-1]
    logp = log_softmax(logits.reshape(-1, num_classes), dim=-1)  # [N, C]
    target = target.reshape(-1)  # [N]
    valid = target != ignore_index
    # gather needs a real class id at every position, so park ignored ones on class 0.
    safe = torch.where(valid, target, torch.zeros_like(target))
    picked = logp.gather(dim=1, index=safe[:, None]).squeeze(1)  # [N]: log p(target)
    nll = -picked * valid
    # 0/0 = NaN when every position is ignored, which is also what F.cross_entropy returns.
    return nll.sum() / valid.sum()
