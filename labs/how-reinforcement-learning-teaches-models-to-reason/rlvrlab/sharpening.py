"""Section 9 of the post: RL raises pass@1 and can lower pass@k at large k.

A toy "base model" picks one of K solution strategies with a softmax over shared logits (it does not look at the
problem). Strategy j solves problem i with probability M[i, j]. Strategy 0 is the common method: it works on most
problems. Every problem also has one rare strategy that works on it. Some problems are solved only by their rare
strategy, and a few by nothing.

pass@k is exact: with per-sample success s_i = sum_j pi_j M[i, j], pass@k_i = 1 - (1 - s_i)^k.
Training is GRPO on sampled groups with 0/1 rewards, plus an exact KL(pi || pi_ref) penalty.
This is a mechanism demo. It does not reproduce Yue et al. or ProRL, which used real models.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import torch

from .advantages import group_advantages


@dataclass
class SharpenConfig:
    strategies: int = 16
    problems: int = 2000
    frac_common: float = 0.6  # problems the common strategy solves
    frac_unsolvable: float = 0.05
    solve_prob: float = 0.8  # chance a working strategy gives the right answer
    rare_logit: float = -1.0  # base logit of each rare strategy (common one is 0)
    steps: int = 300
    batch: int = 64
    group: int = 8
    lr: float = 0.05
    beta: float = 0.0  # KL coefficient
    seed: int = 0


def make_problems(cfg: SharpenConfig, seed: int) -> torch.Tensor:
    """M: (problems, strategies) success probabilities."""
    gen = torch.Generator().manual_seed(seed)
    n, k = cfg.problems, cfg.strategies
    m = torch.zeros(n, k)
    u = torch.rand(n, generator=gen)
    rare = torch.randint(1, k, (n,), generator=gen)
    solvable = u >= cfg.frac_unsolvable
    m[torch.arange(n), rare] = cfg.solve_prob * solvable
    m[:, 0] = cfg.solve_prob * (solvable & (torch.rand(n, generator=gen) < cfg.frac_common))
    return m


def base_logits(cfg: SharpenConfig) -> torch.Tensor:
    theta = torch.full((cfg.strategies,), cfg.rare_logit)
    theta[0] = 0.0
    return theta


def pass_at_k_exact(theta: torch.Tensor, m: torch.Tensor, ks: list[int]) -> dict[int, float]:
    s = m @ theta.softmax(-1)
    return {k: float((1 - (1 - s) ** k).mean()) for k in ks}


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Chen et al. (2021) estimator from n samples with c correct: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def train(cfg: SharpenConfig, m_train: torch.Tensor, eval_m: torch.Tensor, ks: list[int], every: int = 50):
    gen = torch.Generator().manual_seed(cfg.seed)
    ref = base_logits(cfg)
    theta = ref.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=cfg.lr)
    log = []
    for step in range(cfg.steps + 1):
        if step % every == 0:
            with torch.no_grad():
                pi = theta.softmax(-1)
                ent = float(-(pi * pi.log()).sum())
            log.append({"step": step, "entropy": ent, "pi_common": float(pi[0]),
                        **{f"pass@{k}": v for k, v in pass_at_k_exact(theta.detach(), eval_m, ks).items()}})
        if step == cfg.steps:
            break
        idx = torch.randint(0, m_train.shape[0], (cfg.batch,), generator=gen)
        pi = theta.softmax(-1)
        a = torch.multinomial(pi.detach().expand(cfg.batch, -1), cfg.group, replacement=True, generator=gen)
        r = (torch.rand(cfg.batch, cfg.group, generator=gen) < m_train[idx[:, None], a]).float()
        adv = group_advantages(r, "grpo")
        logp = pi.log()[a]
        loss = -(adv * logp).mean()
        if cfg.beta > 0:
            loss = loss + cfg.beta * (pi * (pi.log() - ref.log_softmax(-1))).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return theta.detach(), log
