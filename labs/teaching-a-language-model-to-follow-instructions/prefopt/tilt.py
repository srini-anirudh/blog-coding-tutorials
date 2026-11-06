"""The KL-regularised objective and its closed-form solution, on a space small enough to enumerate.

    max_p  E_p[r] - beta * KL(p || q)          (q = pi_ref, for one prompt)
    p*(y) = q(y) exp(r(y) / beta) / Z,          Z = sum_y q(y) exp(r(y) / beta),   optimum value = beta log Z

Everything works in log space, so a reference probability of exactly zero is log q = -inf.
"""

from __future__ import annotations

import torch


def optimal_logp(ref_logp: torch.Tensor, r: torch.Tensor, beta: float) -> torch.Tensor:
    """log p* = log q + r / beta - log Z. Works on (..., N) with one distribution per row."""
    return torch.log_softmax(ref_logp + r / beta, dim=-1)


def log_partition(ref_logp: torch.Tensor, r: torch.Tensor, beta: float) -> torch.Tensor:
    """log Z(x) = logsumexp(log q + r / beta). Depends on the prompt only."""
    return torch.logsumexp(ref_logp + r / beta, dim=-1)


def kl(p_logp: torch.Tensor, q_logp: torch.Tensor) -> torch.Tensor:
    """KL(p || q) with the convention 0 * log(0 / q) = 0. Infinite if p puts mass where q has none."""
    p = p_logp.exp()
    terms = torch.where(p > 0, p * (p_logp - q_logp), torch.zeros_like(p))
    return terms.sum(-1)


def objective(p_logp: torch.Tensor, ref_logp: torch.Tensor, r: torch.Tensor, beta: float) -> torch.Tensor:
    """J(p) = E_p[r] - beta KL(p || q)."""
    p = p_logp.exp()
    return (p * r).sum(-1) - beta * kl(p_logp, ref_logp)


def solve_by_gradient(ref_logp: torch.Tensor, r: torch.Tensor, beta: float, steps: int = 3000,
                      lr: float = 0.05, seed: int = 0) -> torch.Tensor:
    """Maximise J over free softmax logits with Adam, starting from random logits. Returns log p."""
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(ref_logp.shape, generator=gen, dtype=torch.float64).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        loss = -objective(torch.log_softmax(logits, -1), ref_logp, r, beta).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.log_softmax(logits.detach(), -1)


def implicit_reward(policy_logp: torch.Tensor, ref_logp: torch.Tensor, beta: float) -> torch.Tensor:
    """r_hat = beta log(pi / pi_ref). For pi = p*, this equals r - beta log Z(x): the reward up to a constant."""
    return beta * (policy_logp - ref_logp)


def fit_dpo_tabular(ref_logp: torch.Tensor, prompt: torch.Tensor, winners: torch.Tensor, losers: torch.Tensor,
                    beta: float, steps: int = 3000, lr: float = 0.05) -> torch.Tensor:
    """DPO with one free logit per (prompt, completion), starting at the reference. Returns log pi, (P, K).

    The loss is -log sigmoid(beta [log pi(y_w|x)/ref(y_w|x) - log pi(y_l|x)/ref(y_l|x)]). No reward model anywhere.
    """
    logits = ref_logp.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr / beta)  # the implicit reward is beta * logits, so step in its units
    for _ in range(steps):
        logp = torch.log_softmax(logits, -1)
        r_hat = beta * (logp - ref_logp)
        loss = -torch.nn.functional.logsigmoid(r_hat[prompt, winners] - r_hat[prompt, losers]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.log_softmax(logits.detach(), -1)
