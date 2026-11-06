"""Compounding error: behaviour cloning costs O(eps H^2), training on your own states costs O(eps H).

The corridor. The state is an offset d from the expert's path (d = 0 is on the path) plus the time step. Each step
the learner picks left, stay or right. With probability eps its action is replaced by a mistake that pushes it one
step further off the path: eps is the per-step error rate of the theory. Cost is 1 for every step that ends off the
path. The expert never errs, so its demonstrations contain only d = 0.

Learners are lookup tables over the states they were trained on. An unseen state gets the most common training
label. That is all "nothing has been promised about states off that distribution" means, made concrete.

    BC      : table fit on expert rollouts. Knows only d = 0 -> "stay", so after a mistake it stays off the path.
    DAgger  : roll the learner out, ask the expert to label every state it visited, refit (Ross et al., 2011).
              It sees d = +-1 and learns to step back.
With ``irreversible`` > 0 a mistake instead breaks the world with that probability: no action returns to the path.
"""

from __future__ import annotations

import random
from collections import Counter

LEFT, STAY, RIGHT = -1, 0, 1
BROKEN = None  # the state after an irreversible mistake


def exact_cost(eps: float, H: int) -> float:
    """Ross-Bagnell sum for an absorbing first mistake: sum_t eps (1-eps)^t (H - t)."""
    return sum(eps * (1 - eps) ** t * (H - t) for t in range(H))


def quadratic_approx(eps: float, H: int) -> float:
    return eps * H * (H + 1) / 2


def expert(d) -> int:
    return STAY if d in (0, BROKEN) else (LEFT if d > 0 else RIGHT)


class TablePolicy:
    def __init__(self, labels: dict):
        self.table = {d: Counter(v).most_common(1)[0][0] for d, v in labels.items()}
        self.default = Counter(a for v in labels.values() for a in v).most_common(1)[0][0]

    def __call__(self, d) -> int:
        return self.table.get(d, self.default)


def rollout(policy, H: int, eps: float, rng: random.Random, irreversible: float = 0.0):
    """Returns (cost, visited states). A mistake moves one step away from the path (or breaks the world)."""
    d, cost, visited = 0, 0, []
    for _ in range(H):
        visited.append(d)
        if d is BROKEN:
            cost += 1
            continue
        if rng.random() < eps:
            if rng.random() < irreversible:
                d = BROKEN
            else:
                d = d + (1 if d > 0 else -1 if d < 0 else rng.choice((-1, 1)))
        else:
            d = d + policy(d)
        cost += d != 0
    return cost, visited


def fit_bc(H: int, n_demos: int, rng: random.Random) -> TablePolicy:
    labels: dict = {}
    for _ in range(n_demos):
        _, states = rollout(expert, H, 0.0, rng)
        for d in states:
            labels.setdefault(d, []).append(expert(d))
    return TablePolicy(labels)


def fit_dagger(H: int, eps: float, n_demos: int, iters: int, rng: random.Random,
               irreversible: float = 0.0) -> TablePolicy:
    labels: dict = {}
    for _ in range(n_demos):
        _, states = rollout(expert, H, 0.0, rng)
        for d in states:
            labels.setdefault(d, []).append(expert(d))
    policy = TablePolicy(labels)
    for _ in range(iters):
        for _ in range(n_demos):
            _, states = rollout(policy, H, eps, rng, irreversible)   # the learner's own state distribution
            for d in states:
                labels.setdefault(d, []).append(expert(d))           # labelled by the expert
        policy = TablePolicy(labels)
    return policy


def mean_cost(policy, H: int, eps: float, episodes: int, seed: int, irreversible: float = 0.0) -> float:
    rng = random.Random(seed)
    return sum(rollout(policy, H, eps, rng, irreversible)[0] for _ in range(episodes)) / episodes
