"""Section 5 of the post: group baselines, and what they do when the reward is 0 or 1.

Rewards are laid out (P, G): P prompts, G samples of each. C06 (`c06-post-training/posttrain/grpo.py`) has the
same GRPO advantage inside a full LLM training loop; this file keeps every variant the post compares side by side.
"""

from __future__ import annotations

import math

import torch

KINDS = ("reinforce", "mean", "rloo", "grpo")


def group_advantages(r: torch.Tensor, kind: str, eps: float = 1e-4, unbiased_std: bool = False) -> torch.Tensor:
    """r: (P, G) rewards. Returns (P, G) advantages.

    reinforce: A = r                        (no baseline)
    mean:      A = r - mean(r)              (Dr. GRPO's advantage)
    rloo:      A = r - mean of the other G-1 (leave-one-out, unbiased)
    grpo:      A = (r - mean(r)) / (std(r) + eps)
    """
    g = r.shape[-1]
    mean = r.mean(dim=-1, keepdim=True)
    if kind == "reinforce":
        return r.clone()
    if kind == "mean":
        return r - mean
    if kind == "rloo":
        return (r - mean) * g / (g - 1)  # r_i - (sum_j r_j - r_i)/(G-1), rearranged
    if kind == "grpo":
        return (r - mean) / (r.std(dim=-1, keepdim=True, unbiased=unbiased_std) + eps)
    raise ValueError(kind)


def binary_closed_form(p: float) -> tuple[float, float]:
    """GRPO advantages when a fraction p of the group is right (population std): (+sqrt((1-p)/p), -sqrt(p/(1-p)))."""
    return math.sqrt((1 - p) / p), -math.sqrt(p / (1 - p))


def informative_prob(p: float, g: int) -> float:
    """Chance that a group of g samples, each right with probability p, is not unanimous (so it has a gradient)."""
    return 1.0 - p**g - (1.0 - p) ** g


def dynamic_sampling_mask(r: torch.Tensor) -> torch.Tensor:
    """DAPO's filter: keep only groups whose rewards are not all equal. r: (P, G) -> (P,) bool."""
    return r.max(dim=-1).values > r.min(dim=-1).values


def advantage_mass(k: int, g: int, kind: str) -> float:
    """Sum of |A_i| over a group of g binary rewards with k right: how hard this prompt pushes on the policy."""
    r = torch.zeros(1, g)
    r[0, :k] = 1.0
    return float(group_advantages(r, kind, eps=0.0 if 0 < k < g else 1e-4).abs().sum())
