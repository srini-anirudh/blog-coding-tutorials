"""Group-relative RL on a combination lock, to watch a batch go empty as the horizon grows.

The task. A lock with H positions and K choices per position. Each task has its own secret code. At position t the
agent picks one of K actions; a wrong pick ends the episode. Binary reward: 1 if all H picks were right. Graded reward:
the fraction of positions passed before the first wrong pick.

The policy. A table of logits theta[task, position, action] (a softmax policy, trained with autograd). It starts
"warm": every position puts probability q0 on the right action, which stands in for a policy after behaviour
cloning. With q0 = 0.8 and H = 40, p = 0.8^40, about 1.3e-4.

The update is the trajectory-level group-relative estimator: G rollouts per task, A_i = (R_i - mean) / (std + eps),
and every action in rollout i gets A_i. Groups with zero reward variance contribute exactly zero.

Interventions from the post, each a flag:
    graded=True      partial credit instead of a binary reward
    curriculum       a schedule h_t = min(H_max, h0 + delta * floor(t / T_s)) on the task horizon: the reward
                     only asks for the first h_t positions to be right
    filter_zero=True drop zero-variance groups before the backward pass (saves compute; same gradient direction)
    G                group size
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class LockConfig:
    H: int = 40                       # target horizon
    K: int = 4                        # choices per position
    n_tasks: int = 16
    G: int = 8
    q0: float = 0.8                   # initial probability of the right action at every position
    lr: float = 20.0
    iters: int = 300
    graded: bool = False
    filter_zero: bool = False
    curriculum: tuple[int, int, int] | None = None   # (h0, delta, T_s)
    seed: int = 0


def horizon_at(cfg: LockConfig, it: int) -> int:
    if cfg.curriculum is None:
        return cfg.H
    h0, delta, Ts = cfg.curriculum
    return min(cfg.H, h0 + delta * (it // Ts))


def init_logits(cfg: LockConfig, codes: torch.Tensor) -> torch.Tensor:
    """Logits so that softmax puts q0 on the code and (1 - q0) / (K - 1) on each other action."""
    gap = math.log(cfg.q0 * (cfg.K - 1) / (1 - cfg.q0))
    logits = torch.zeros(cfg.n_tasks, cfg.H, cfg.K)
    logits.scatter_(2, codes.unsqueeze(-1), gap)
    return logits.requires_grad_(True)


def rollout(logits: torch.Tensor, codes: torch.Tensor, G: int, h: int, gen: torch.Generator):
    """G rollouts per task over the first h positions.

    Returns actions (n_tasks, G, h), alive mask (n_tasks, G, h) marking positions actually played, and passed
    (n_tasks, G), the number of positions passed before the first wrong pick.
    """
    probs = torch.softmax(logits[:, :h].detach(), -1)                    # (n, h, K)
    n = probs.shape[0]
    flat = probs.unsqueeze(1).expand(n, G, h, -1).reshape(-1, probs.shape[-1])
    actions = torch.multinomial(flat, 1, generator=gen).view(n, G, h)
    right = actions == codes[:, None, :h]
    passed = torch.where(right.all(-1), torch.full((n, G), h), (~right).float().argmax(-1))
    alive = torch.arange(h)[None, None, :] <= passed[..., None]          # the wrong pick itself is played
    alive &= torch.arange(h)[None, None, :] < h
    return actions, alive, passed


def advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(n_tasks, G) rewards -> (n_tasks, G) group-relative advantages. Zero-variance groups give exactly 0."""
    mean = rewards.mean(1, keepdim=True)
    std = rewards.std(1, keepdim=True)
    return (rewards - mean) / (std + eps)


@dataclass
class LockLog:
    rows: list[dict] = field(default_factory=list)

    def last(self, key: str, k: int = 10) -> float:
        xs = [r[key] for r in self.rows[-k:]]
        return sum(xs) / len(xs)


def sampled_success(logits: torch.Tensor, codes: torch.Tensor, h: int) -> float:
    """Exact expected binary success of the sampling policy: product of per-position right-action probabilities."""
    p = torch.softmax(logits[:, :h].detach(), -1).gather(2, codes[:, :h, None]).squeeze(-1)
    return float(p.prod(-1).mean())


def train(cfg: LockConfig) -> tuple[LockLog, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(cfg.seed)
    codes = torch.randint(0, cfg.K, (cfg.n_tasks, cfg.H), generator=gen)
    logits = init_logits(cfg, codes)
    opt = torch.optim.SGD([logits], lr=cfg.lr)
    log = LockLog()
    env_steps = 0
    for it in range(cfg.iters):
        h = horizon_at(cfg, it)
        actions, alive, passed = rollout(logits, codes, cfg.G, h, gen)
        env_steps += int(alive.sum())
        success = (passed == h).float()
        rewards = passed.float() / h if cfg.graded else success
        adv = advantages(rewards)
        informative = rewards.std(1) > 0
        keep = informative if cfg.filter_zero else torch.ones_like(informative)
        logp = torch.log_softmax(logits[:, :h], -1)                        # (n, h, K)
        logp_a = logp.unsqueeze(1).expand(-1, cfg.G, -1, -1).gather(3, actions.unsqueeze(-1)).squeeze(-1)
        mask = alive & keep[:, None, None]
        n_rollouts = int(keep.sum()) * cfg.G
        opt.zero_grad()
        if n_rollouts:
            loss = -(adv[..., None] * logp_a * mask).sum() / (cfg.n_tasks * cfg.G)
            loss.backward()
            opt.step()
        log.rows.append({"iter": it, "h": h, "informative": float(informative.float().mean()),
                         "batch_success": float(success.mean()), "backward_rollouts": n_rollouts,
                         "env_steps": env_steps,
                         "target_sampled_success": sampled_success(logits, codes, cfg.H)})
    return log, logits.detach(), codes
