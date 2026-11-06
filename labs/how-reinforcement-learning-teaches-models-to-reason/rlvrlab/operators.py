"""Section 10 of the post: SFT and RL as two operators on the same distribution.

The policy is a softmax over K modes (whole solution strategies). One mode, `target`, is the demonstration. It
starts almost outside the support (probability ~1e-6).

  SFT gradient        E_{y~D}[grad log pi(y)]  =  E_{y~pi}[ (D(y)/pi(y)) grad log pi(y) ]
  implicit reward     r(y) = D(y) / pi(y): 1/pi on the demonstration, 0 elsewhere
  rectified SFT (DFT) multiply the per-token loss by a detached pi(y): the implicit reward becomes D(y), a constant
  RL with a verifier  E_{y~pi}[ A(y) grad log pi(y) ], A signed and relative to the group
"""

from __future__ import annotations

import torch

from .advantages import group_advantages


def sft_grad_exact(theta: torch.Tensor, target: int) -> torch.Tensor:
    """Ascent direction of log pi(target) with respect to the logits: onehot(target) - pi."""
    return torch.nn.functional.one_hot(torch.tensor(target), theta.numel()).to(theta.dtype) - theta.softmax(-1)


def sft_grad_as_policy_gradient(theta: torch.Tensor, target: int, n: int, gen: torch.Generator) -> torch.Tensor:
    """The same gradient, estimated on-policy with the implicit reward D(y)/pi(y). Unbiased, and very noisy."""
    pi = theta.softmax(-1)
    y = torch.multinomial(pi, n, replacement=True, generator=gen)
    hits = (y == target).sum().to(theta.dtype)  # only samples equal to the target carry reward 1/pi(target)
    onehot = torch.nn.functional.one_hot(torch.tensor(target), theta.numel()).to(theta.dtype)
    return (hits / n) / pi[target] * (onehot - pi)


def token_weights(p: torch.Tensor) -> dict[str, torch.Tensor]:
    """For a demonstration token the model gives probability p: implicit reward and logit-gradient norm, SFT vs DFT.

    SFT loss -log p has logit gradient (pi - onehot). With the other mass on one competitor, its norm is sqrt(2)(1-p).
    DFT loss -sg(p) log p scales that by p.
    """
    sft_norm = (2.0**0.5) * (1 - p)
    return {"implicit_reward_sft": 1 / p, "implicit_reward_dft": torch.ones_like(p),
            "grad_norm_sft": sft_norm, "grad_norm_dft": p * sft_norm}


def run_operator(theta0: torch.Tensor, op: str, target: int, rewarded: torch.Tensor, steps: int, lr: float,
                 group: int = 8, seed: int = 0) -> tuple[torch.Tensor, int]:
    """Apply `steps` plain gradient-ascent steps of one operator. Returns (final logits, times target was sampled).

    op = "sft": ascend log pi(target).
    op = "rl":  GRPO on one prompt, reward 1 for modes in `rewarded` (a 0/1 vector over modes).
    """
    gen = torch.Generator().manual_seed(seed)
    theta = theta0.clone()
    seen = 0
    for _ in range(steps):
        pi = theta.softmax(-1)
        if op == "sft":
            g = sft_grad_exact(theta, target)
        elif op == "rl":
            y = torch.multinomial(pi, group, replacement=True, generator=gen)
            seen += int((y == target).sum())
            adv = group_advantages(rewarded[y][None, :], "grpo")[0]
            onehot = torch.nn.functional.one_hot(y, theta.numel()).to(theta.dtype)
            g = (adv[:, None] * (onehot - pi)).mean(0)
        else:
            raise ValueError(op)
        theta = theta + lr * g
    return theta, seen


def one_step_ratios(theta: torch.Tensor, g: torch.Tensor, lr: float) -> torch.Tensor:
    """pi_new / pi_old for every mode after one step along g."""
    return (theta + lr * g).softmax(-1) / theta.softmax(-1)
