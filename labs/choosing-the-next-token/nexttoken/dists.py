"""The logit vector and the numbers we read off it: softmax, entropy, temperature.

All functions take logits of shape (..., V) and work on the last axis. Entropy is in nats unless the name says bits.
"""

from __future__ import annotations

import math

import torch

LN2 = math.log(2.0)


def softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """p_i = exp(z_i / T) / sum_j exp(z_j / T). Subtracting the max first is the shift invariance, used for stability."""
    z = logits.float() / temperature
    z = z - z.max(dim=-1, keepdim=True).values
    e = z.exp()
    return e / e.sum(dim=-1, keepdim=True)


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """H(p) = -sum p log p in nats. Zero-probability entries contribute 0, not nan."""
    logp = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
    return -(probs * logp).sum(dim=-1)


def entropy_bits(probs: torch.Tensor) -> torch.Tensor:
    return entropy(probs) / LN2


def effective_support(probs: torch.Tensor) -> torch.Tensor:
    """2^H in bits, which equals e^H in nats: the number of equally likely options with the same entropy."""
    return entropy(probs).exp()


def dH_dT(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """The closed form dH/dT = Var_{p(T)}[z] / T^3, entropy in nats."""
    p = softmax(logits, temperature)
    z = logits.float()
    mean = (p * z).sum(-1)
    var = (p * z * z).sum(-1) - mean**2
    return var / temperature**3


def odds_ratio(logits: torch.Tensor, temperature: float) -> float:
    """p_2 / p_1 for the top two tokens. Equals exp(-gap / T)."""
    p = softmax(logits, temperature)
    top = torch.topk(p, 2).values
    return (top[1] / top[0]).item()
