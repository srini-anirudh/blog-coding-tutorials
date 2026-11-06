"""Stability tricks for attention logits: QK normalisation and soft-capping."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """x / rms(x) * gain over the last dim. Used as the per-head Q and K norm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * self.weight.float()).to(x.dtype)


def soft_cap(scores: torch.Tensor, cap: float) -> torch.Tensor:
    """s -> cap * tanh(s / cap). Near-identity for |s| << cap, never beyond +-cap."""
    return cap * torch.tanh(scores / cap)


def logit_stats(scores: torch.Tensor) -> dict[str, float]:
    """Summaries of a (..., Nq, Nk) logit tensor: how large, and how peaked the softmax is."""
    probs = torch.softmax(scores.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
    return {
        "max_logit": scores.abs().max().item(),
        "logit_std": scores.float().std().item(),
        "mean_max_prob": probs.amax(-1).mean().item(),
        "mean_entropy": entropy.mean().item(),
    }
