"""The KL-regularised objective on a finite set of completions, and its closed-form optimum.

    J(p) = E_p[r] - beta * KL(p || q)        maximised by    p*(y) = q(y) exp(r(y) / beta) / Z

With a finite set we can compute Z exactly, which a real language model never can.
"""

from __future__ import annotations

import torch


def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) with the convention 0 log 0 = 0. Infinite if p puts mass where q has none."""
    p, q = p.double(), q.double()
    on = p > 0
    if bool((q[on] == 0).any()):
        return torch.tensor(float("inf"), dtype=torch.float64)
    return (p[on] * (p[on].log() - q[on].log())).sum()


def objective(p: torch.Tensor, q: torch.Tensor, r: torch.Tensor, beta: float) -> torch.Tensor:
    return (p.double() * r.double()).sum() - beta * kl(p, q)


def tilt(q: torch.Tensor, r: torch.Tensor, beta: float) -> torch.Tensor:
    """The exponential tilt q * exp(r / beta) / Z, computed in log space. Zero-probability items stay at zero."""
    logits = q.double().log() + r.double() / beta  # log 0 = -inf survives the softmax as exactly 0
    return torch.softmax(logits, dim=-1)


def log_partition(q: torch.Tensor, r: torch.Tensor, beta: float) -> torch.Tensor:
    """log Z = log sum_y q(y) exp(r(y) / beta)."""
    return torch.logsumexp(q.double().log() + r.double() / beta, dim=-1)


def maximise_numerically(q: torch.Tensor, r: torch.Tensor, beta: float, steps: int = 4000, lr: float = 0.05):
    """Gradient ascent on J over softmax logits, with no knowledge of the closed form.

    Items with q = 0 are masked out: any mass there would make the KL infinite.
    """
    support = q > 0
    z = torch.zeros(q.numel(), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        p = torch.softmax(z.masked_fill(~support, float("-inf")), -1)
        loss = -objective(p, q, r, beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(z.detach().masked_fill(~support, float("-inf")), -1)


def best_of_n(q: torch.Tensor, score: torch.Tensor, n: int) -> torch.Tensor:
    """Exact distribution of best-of-n sampling from q, ranked by `score` (ties broken by index).

    Sort by score. If F(k) is the mass of the k lowest-scored items, the chance that the best of n draws is item k
    is F(k)^n - F(k-1)^n.
    """
    order = torch.argsort(score.double() + 1e-12 * torch.arange(score.numel(), dtype=torch.float64))
    cdf = torch.cumsum(q.double()[order], 0).clamp(max=1.0)
    prev = torch.cat([torch.zeros(1, dtype=torch.float64), cdf[:-1]])
    p_sorted = cdf**n - prev**n
    p = torch.empty_like(p_sorted)
    p[order] = p_sorted
    return p
