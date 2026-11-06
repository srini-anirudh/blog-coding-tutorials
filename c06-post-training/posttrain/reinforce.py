"""Stage 4: REINFORCE, the simplest policy gradient, and a measurement of why it is noisy.

    grad J = E[ (R - b) * grad log pi(y | x) ],   log pi(y | x) = sum over completion tokens

With a 0/1 reward and no baseline (b = 0), every wrong sample contributes nothing and every right sample pushes
up whatever it happened to say. Baselines change the variance of the estimate, not its expectation.
"""

from __future__ import annotations

import torch

from .grpo import group_advantages


def reinforce_loss(logps: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """-(weight * log pi(y|x)).mean() over sequences.

    logps, mask: (N, C) per-token log-probs and completion mask.  weights: (N,) = R - b, no grad.
    The sum over tokens (not the mean) is what makes this the true gradient of log pi(y|x).
    """
    seq_logp = (logps * mask.to(logps.dtype)).sum(-1)  # (N,)
    return -(weights.detach() * seq_logp).mean()


def estimator_weights(rewards: torch.Tensor, group_size: int, estimator: str) -> torch.Tensor:
    """Per-sample weights for the four estimators compared in scripts/reinforce_variance.py.

    "none":      R                     (vanilla REINFORCE)
    "batch":     R - mean over batch   (one shared baseline)
    "group":     R - mean over the G samples of the same prompt (RLOO-style, without leave-one-out)
    "group_std": (R - group mean) / group std  (GRPO advantages)
    """
    if estimator == "none":
        return rewards
    if estimator == "batch":
        return rewards - rewards.mean()
    if estimator == "group":
        return group_advantages(rewards, group_size, scale="none")
    if estimator == "group_std":
        return group_advantages(rewards, group_size, scale="group")
    raise ValueError(estimator)


def flat_grad(params: list[torch.nn.Parameter]) -> torch.Tensor:
    """Concatenate .grad of `params` into one vector (zeros where a grad is missing)."""
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])


def gradient_noise(grads: torch.Tensor) -> dict[str, float]:
    """grads: (K, D), K independent estimates of the same gradient.

    Returns the signal-to-noise ratio ||mean||^2 / E||g - mean||^2 and the mean pairwise cosine similarity.
    Pairwise cosine near 0 means two independent batches disagree about which way to move.
    """
    mean = grads.mean(0)
    noise = ((grads - mean) ** 2).sum(-1).mean()
    unit = grads / grads.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    cos = unit @ unit.T
    K = grads.shape[0]
    off_diag = (cos.sum() - cos.diagonal().sum()) / (K * (K - 1))
    return {
        "snr": float(mean.pow(2).sum() / noise.clamp(min=1e-30)),
        "pairwise_cos": float(off_diag),
        "mean_norm": float(grads.norm(dim=-1).mean()),
    }
