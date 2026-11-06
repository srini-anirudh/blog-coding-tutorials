"""Section 5 of the post: policy gradients on a contextual bandit, where the exact gradient is known.

One prompt, K possible whole responses ("actions"), a softmax policy over logits theta, and a 0/1 verifier
that accepts action k with probability c_k. J(theta) = sum_k pi_k c_k, so the exact gradient with respect to the
logits is pi * (c - J). Every estimator below can be checked against it.
"""

from __future__ import annotations

import torch

from .advantages import group_advantages


def exact_gradient(theta: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    pi = theta.softmax(-1)
    return pi * (c - (pi * c).sum())


def sample_groups(theta: torch.Tensor, c: torch.Tensor, groups: int, g: int, gen: torch.Generator):
    """Draw `groups` independent groups of g actions and their 0/1 rewards. Returns (actions, rewards), (groups, g)."""
    pi = theta.softmax(-1)
    actions = torch.multinomial(pi.expand(groups, -1), g, replacement=True, generator=gen)
    rewards = (torch.rand(groups, g, generator=gen) < c[actions]).to(theta.dtype)
    return actions, rewards


def group_gradients(theta: torch.Tensor, actions: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    """One gradient estimate per group: (1/G) sum_i A_i * grad log pi(a_i), with grad log pi(a) = onehot(a) - pi."""
    pi = theta.softmax(-1)
    g = actions.shape[1]
    pushed = torch.zeros(actions.shape[0], theta.numel(), dtype=theta.dtype).scatter_add_(1, actions, adv)
    return (pushed - adv.sum(1, keepdim=True) * pi) / g  # sum_i A_i onehot(a_i) - (sum_i A_i) pi, over G


def estimator_stats(theta, c, kind: str, g: int, groups: int = 200_000, seed: int = 0, baseline: float | None = None):
    """Monte Carlo mean and spread of one estimator.

    Returns the ratio of the mean estimate's projection on the exact gradient to its norm (1.0 = unbiased scale),
    the cosine with the exact gradient, the bias norm relative to the exact gradient norm, and the total variance
    (trace of the covariance of a single-group estimate), in units of the exact gradient's squared norm.
    """
    gen = torch.Generator().manual_seed(seed)
    actions, rewards = sample_groups(theta, c, groups, g, gen)
    adv = rewards - baseline if baseline is not None else group_advantages(rewards, kind)
    est = group_gradients(theta, actions, adv)
    true = exact_gradient(theta, c)
    mean = est.mean(0)
    tn = true.norm()
    return {
        "scale": float((mean @ true) / tn**2),
        "cosine": float(torch.nn.functional.cosine_similarity(mean, true, dim=0)),
        "rel_bias": float((mean - true).norm() / tn),
        "rel_variance": float(est.var(0).sum() / tn**2),
        "mc_stderr": float((est.var(0).sum() / groups).sqrt() / tn),
    }
