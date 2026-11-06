"""Credit assignment: one scalar per trajectory (GRPO) against step-level groups at anchor states (GiGPO).

The environment is a chain of ``stages``. At every stage the agent has ``actions`` choices, and their roles are
hidden and differ per task: one advances, one is a dead end (the episode ends, reward 0), the rest are no-ops
(the world does not change, a step is wasted). The episode succeeds, reward 1, if it reaches the last stage within
``horizon`` steps. Reward is terminal only.

No-ops are what make anchor states appear. A trajectory that wastes a step stands in the same state twice, and
sibling trajectories of the same task stand in the same states too. GiGPO groups every action taken from the same
state across the group and compares their discounted returns:

    A_E(tau_i)   = (R(tau_i) - mean_G R) / std_G R                       one number per trajectory
    A_S(a_t^(i)) = (R_t^(i) - mean_{G^S(s)} R_t) / std_{G^S(s)} R_t,     R_t = gamma^(T-1-t) * R  (terminal reward)
    A            = A_E + omega * A_S

``state_noise`` is the probability that the harness fails to recognise a state as one it has seen (a timestamp in
the rendered page, say). Then that step's anchor group has size 1 and A_S = 0. The policy still sees the true state.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

import torch

ADVANCE, DEAD_END, NOOP = "advance", "dead_end", "noop"


@dataclass
class ChainConfig:
    n_tasks: int = 64
    stages: int = 4
    actions: int = 4
    horizon: int = 12
    gamma: float = 0.9


@dataclass
class Episode:
    task: int
    states: list = field(default_factory=list)     # stage before each action
    actions: list = field(default_factory=list)
    keys: list = field(default_factory=list)       # what the harness's state matcher sees
    reward: float = 0.0


def make_roles(cfg: ChainConfig, seed: int) -> list[list[list[str]]]:
    """roles[task][stage][action]."""
    rng = random.Random(seed)
    base = [ADVANCE, DEAD_END] + [NOOP] * (cfg.actions - 2)
    return [[rng.sample(base, len(base)) for _ in range(cfg.stages)] for _ in range(cfg.n_tasks)]


def run_episode(task: int, probs: torch.Tensor, roles, cfg: ChainConfig, rng: random.Random,
                state_noise: float = 0.0) -> Episode:
    ep, stage = Episode(task), 0
    for _ in range(cfg.horizon):
        a = rng.choices(range(cfg.actions), weights=probs[task, stage].tolist())[0]
        ep.states.append(stage)
        ep.actions.append(a)
        ep.keys.append((task, stage) if rng.random() >= state_noise else (task, stage, rng.random()))
        role = roles[task][stage][a]
        if role == DEAD_END:
            break
        if role == ADVANCE:
            stage += 1
            if stage == cfg.stages:
                ep.reward = 1.0
                break
    return ep


def _normalise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if len(x) < 2:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std() + eps)


def episode_advantages(group: list[Episode]) -> list[float]:
    """GRPO: one advantage per trajectory, shared by all its actions."""
    return _normalise(torch.tensor([e.reward for e in group])).tolist()


def step_returns(ep: Episode, gamma: float) -> list[float]:
    T = len(ep.actions)
    return [gamma ** (T - 1 - t) * ep.reward for t in range(T)]


def anchor_advantages(group: list[Episode], gamma: float) -> tuple[list[list[float]], float]:
    """GiGPO step level: per-step A_S for each episode, and the share of steps whose anchor group has size >= 2."""
    buckets = defaultdict(list)                     # anchor key -> [(episode index, step)]
    returns = [step_returns(e, gamma) for e in group]
    for i, e in enumerate(group):
        for t, k in enumerate(e.keys):
            buckets[k].append((i, t))
    adv = [[0.0] * len(e.actions) for e in group]
    shared = 0
    for members in buckets.values():
        if len(members) > 1:
            shared += len(members)
        a = _normalise(torch.tensor([returns[i][t] for i, t in members]))
        for (i, t), v in zip(members, a.tolist()):
            adv[i][t] = v
    return adv, shared / max(sum(len(e.actions) for e in group), 1)


def success_probability(probs: torch.Tensor, roles, cfg: ChainConfig) -> torch.Tensor:
    """Exact P(success) per task under the current policy, by dynamic programming. No sampling noise."""
    out = torch.zeros(cfg.n_tasks)
    for task in range(cfg.n_tasks):
        f = [[0.0] * (cfg.horizon + 1) for _ in range(cfg.stages + 1)]
        f[cfg.stages] = [1.0] * (cfg.horizon + 1)
        for left in range(1, cfg.horizon + 1):
            for s in range(cfg.stages):
                p = probs[task, s].tolist()
                r = roles[task][s]
                adv = sum(pi for pi, ro in zip(p, r) if ro == ADVANCE)
                noop = sum(pi for pi, ro in zip(p, r) if ro == NOOP)
                f[s][left] = adv * f[s + 1][left - 1] + noop * f[s][left - 1]
        out[task] = f[0][cfg.horizon]
    return out


def noop_probability(probs: torch.Tensor, roles) -> float:
    vals = [sum(p for p, ro in zip(probs[t, s].tolist(), roles[t][s]) if ro == NOOP)
            for t in range(len(roles)) for s in range(len(roles[0]))]
    return sum(vals) / len(vals)


def train(method: str, iters: int, seed: int, cfg: ChainConfig | None = None, group_size: int = 8,
          tasks_per_iter: int = 16, lr: float = 0.1, omega: float = 1.0, state_noise: float = 0.0,
          eval_at=(10, 25, 50, 100, 200), log=print) -> dict:
    """method: 'grpo' (A_E only) or 'gigpo' (A_E + omega * A_S)."""
    cfg = cfg or ChainConfig()
    roles = make_roles(cfg, seed=0)               # the same task set for every run
    rng = random.Random(seed)
    torch.manual_seed(seed)
    logits = torch.zeros(cfg.n_tasks, cfg.stages, cfg.actions, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    history, coverage = [], []
    for it in range(1, iters + 1):
        probs = torch.softmax(logits.detach(), -1)
        idx, adv = [], []
        for task in rng.sample(range(cfg.n_tasks), tasks_per_iter):
            group = [run_episode(task, probs, roles, cfg, rng, state_noise) for _ in range(group_size)]
            a_e = episode_advantages(group)
            a_s, cov = anchor_advantages(group, cfg.gamma)
            coverage.append(cov)
            for i, e in enumerate(group):
                for t, (s, a) in enumerate(zip(e.states, e.actions)):
                    idx.append((task, s, a))
                    adv.append(a_e[i] + (omega * a_s[i][t] if method == "gigpo" else 0.0))
        t_, s_, a_ = map(torch.tensor, zip(*idx))
        logp = torch.log_softmax(logits[t_, s_], -1).gather(1, a_[:, None]).squeeze(1)
        loss = -(torch.tensor(adv) * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it in eval_at or it == iters:
            probs = torch.softmax(logits.detach(), -1)
            p = success_probability(probs, roles, cfg)
            row = {"iter": it, "success": p.mean().item(), "noop_prob": noop_probability(probs, roles),
                   "anchor_coverage": sum(coverage) / len(coverage), "per_task": p.tolist()}
            history.append(row)
            log(f"{method} noise {state_noise} seed {seed} iter {it} success {row['success']:.3f} "
                f"noop {row['noop_prob']:.3f} coverage {row['anchor_coverage']:.3f}")
    return {"method": method, "state_noise": state_noise, "seed": seed, "history": history}
