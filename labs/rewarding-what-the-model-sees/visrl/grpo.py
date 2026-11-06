"""Group-relative advantages and the zero-advantage problem.

    one image, K rollouts --> rewards R_1..R_K
    A_k = (R_k - mean(R)) / (std(R) + eps)          every token of rollout k gets A_k

If all K rewards are equal, every A_k is 0 and the image teaches nothing.
For a binary reward with pass rate p that happens with probability p^K + (1-p)^K.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, scale_by_std: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """rewards (P, K) -> advantages (P, K). The baseline is the mean over rollouts of the SAME image.

    std is the population std (divide by K). scale_by_std=False is the Dr. GRPO variant:
    near-ties are then not blown up into large advantages.
    """
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    centered = torch.where(zero_advantage(rewards)[..., None], torch.zeros_like(centered), centered)
    if not scale_by_std:
        return centered
    return centered / (rewards.std(dim=-1, unbiased=False, keepdim=True) + eps)


def zero_advantage(rewards: torch.Tensor, tol: float = 1e-9) -> torch.Tensor:
    """(P, K) -> (P,) bool: True where every rollout in the group got the same reward."""
    return (rewards.max(dim=-1).values - rewards.min(dim=-1).values) <= tol


def p_zero_advantage(p: float | torch.Tensor, k: int):
    """P(all K binary rewards agree) = p^K + (1 - p)^K."""
    return p ** k + (1 - p) ** k


def sample_pass_rates(n: int, mean: float, concentration: float = 4.0) -> torch.Tensor:
    """n prompt pass rates from Beta(mean * c, (1 - mean) * c): a batch whose difficulty is spread around `mean`.

    Uses the global torch seed (torch.distributions takes no generator).
    """
    beta = torch.distributions.Beta(torch.tensor(mean * concentration), torch.tensor((1 - mean) * concentration))
    return beta.sample((n,))


def useful_fraction(pass_rates: torch.Tensor, k: int, generator: torch.Generator) -> float:
    """Sample K binary rewards per prompt and return the share of groups with a non-zero advantage."""
    rewards = (torch.rand(len(pass_rates), k, generator=generator) < pass_rates[:, None]).float()
    return float((~zero_advantage(rewards)).float().mean())


def difficulty_filter(pass_rates: torch.Tensor, low: float = 0.2, high: float = 0.8) -> torch.Tensor:
    """Keep prompts whose estimated pass rate lies in [low, high] (Vision-G1, InternVL3.5)."""
    return pass_rates[(pass_rates >= low) & (pass_rates <= high)]
