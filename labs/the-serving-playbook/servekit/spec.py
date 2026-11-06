"""Speculative sampling: the accept / resample rule, and the expected tokens per target pass.

Leviathan et al. 2022 and Chen et al. 2023. The draft proposes gamma tokens from q. The target scores all
gamma + 1 positions in one pass, giving p. Token x_i is kept with probability min(1, p_i(x_i) / q_i(x_i)).
At the first rejection, a replacement is drawn from norm(max(0, p_i - q_i)). If all gamma survive, one bonus
token is drawn from p_gamma. The emitted tokens are distributed exactly as sampling from p alone.
"""

from __future__ import annotations

import torch


def expected_tokens(alpha: float, gamma: int) -> float:
    """(1 - alpha^(gamma+1)) / (1 - alpha): tokens emitted per target pass when each guess survives w.p. alpha."""
    if alpha == 1.0:
        return gamma + 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def acceptance_rate(p: torch.Tensor, q: torch.Tensor) -> float:
    """alpha = sum_x min(p(x), q(x)) for one position."""
    return torch.minimum(p, q).sum().item()


def speculative_step(p: torch.Tensor, q: torch.Tensor, gen: torch.Generator) -> list[int]:
    """p: (gamma + 1, V) target probabilities. q: (gamma, V) draft probabilities. Returns emitted tokens."""
    gamma = q.shape[0]
    draft = [int(torch.multinomial(q[i], 1, generator=gen)) for i in range(gamma)]
    out: list[int] = []
    for i, x in enumerate(draft):
        u = torch.rand((), generator=gen).item()
        if u < min(1.0, (p[i, x] / q[i, x]).item()):
            out.append(x)
            continue
        residual = torch.clamp(p[i] - q[i], min=0)
        out.append(int(torch.multinomial(residual / residual.sum(), 1, generator=gen)))
        return out
    out.append(int(torch.multinomial(p[gamma], 1, generator=gen)))  # every guess survived: a free bonus token
    return out
