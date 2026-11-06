"""Sparse, hand-shaped and potential-based rewards on a six-page web task.

The world is a deterministic graph of pages. The task is to buy one item:

    home -> search -> results -> product -> cart -> checkout (goal, terminal)

with back links (product -> results, cart -> product, results -> search) and a "reload" self-loop on every page,
because real browsers have both. The step budget is finite, so wasting steps has a price under discounting.

Three reward functions over the same transitions:
    sparse      +1 on reaching checkout. Nothing else.
    milestone   sparse, plus +0.1 every time the agent enters `product` (it found the right page) and +0.1 every
                time it enters `cart` (it put the right item in). A reasonable-looking dense reward.
    potential   sparse, plus F(s, a, s') = gamma * Phi(s') - Phi(s), with Phi = progress along the path
                (Ng, Harada and Russell, 1999). Phi(goal) = 0 because the episode ends there.

``value_iteration`` finds the optimal policy of each reward exactly. ``cycle_gain`` is the post's test: walk every
simple cycle of the graph and add up the shaping reward collected around it. Positive means a policy can farm it.
"""

from __future__ import annotations

import itertools

import numpy as np

PAGES = ["home", "search", "results", "product", "cart", "checkout"]
GOAL = PAGES.index("checkout")
EDGES = {                                     # page -> {action: next page}
    "home": {"open_search": "search", "reload": "home"},
    "search": {"submit_query": "results", "reload": "search"},
    "results": {"click_item": "product", "back": "search", "reload": "results"},
    "product": {"add_to_cart": "cart", "back": "results", "reload": "product"},
    "cart": {"checkout": "checkout", "back": "product", "reload": "cart"},
    "checkout": {},
}
PROGRESS = {p: i / (len(PAGES) - 1) for i, p in enumerate(PAGES)}   # 0 at home, 1 at checkout


def transitions() -> list[tuple[int, str, int]]:
    return [(PAGES.index(s), a, PAGES.index(t)) for s, acts in EDGES.items() for a, t in acts.items()]


def sparse(s: int, s2: int, gamma: float, last: bool = False) -> float:
    return 1.0 if s2 == GOAL else 0.0


def milestone(s: int, s2: int, gamma: float, last: bool = False) -> float:
    bonus = 0.1 if PAGES[s2] in ("product", "cart") and s2 != s else 0.0
    return sparse(s, s2, gamma) + bonus


def make_potential(phi: dict[str, float], zero_at_timeout: bool = True):
    """Sparse plus gamma * Phi(s') - Phi(s). Phi is 0 at the goal, and also on the last step of the budget when
    ``zero_at_timeout`` (a time-out ends the episode too). Without that, a time-out keeps gamma^H Phi(s_H), which
    can change the optimal policy of a finite-horizon task."""
    def reward(s: int, s2: int, gamma: float, last: bool = False) -> float:
        ends = s2 == GOAL or (last and zero_at_timeout)
        phi_next = 0.0 if ends else phi[PAGES[s2]]
        return sparse(s, s2, gamma) + gamma * phi_next - phi[PAGES[s]]
    return reward


potential = make_potential(PROGRESS)
REWARDS = {"sparse": sparse, "milestone": milestone, "potential": potential}


def value_iteration(reward, gamma: float, horizon: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Finite-horizon optimal values V[h, s] (h steps left) and Q-values of the first decision at each state."""
    n = len(PAGES)
    V = np.zeros((horizon + 1, n))
    for h in range(1, horizon + 1):
        for s in range(n):
            if s == GOAL:
                continue
            V[h, s] = max(reward(s, PAGES.index(t), gamma, h == 1) + gamma * V[h - 1, PAGES.index(t)]
                          for t in EDGES[PAGES[s]].values())
    Q = [{a: reward(s, PAGES.index(t), gamma, horizon == 1) + gamma * V[horizon - 1, PAGES.index(t)]
          for a, t in EDGES[PAGES[s]].items()} for s in range(n)]
    return V, Q


def greedy_rollout(reward, gamma: float, horizon: int, start: str = "home") -> tuple[list[str], float, bool]:
    """Follow the optimal finite-horizon policy. Returns (pages visited, return, reached goal)."""
    V, _ = value_iteration(reward, gamma, horizon)
    s, path, ret = PAGES.index(start), [start], 0.0
    for h in range(horizon, 0, -1):
        if s == GOAL:
            break
        best_a, best_t, best_q = None, None, -np.inf
        for a, t in EDGES[PAGES[s]].items():           # ties broken by listing order: forward actions first
            t_i = PAGES.index(t)
            q = reward(s, t_i, gamma, h == 1) + gamma * V[h - 1, t_i]
            if q > best_q + 1e-12:
                best_a, best_t, best_q = a, t_i, q
        ret += gamma ** (horizon - h) * reward(s, best_t, gamma, h == 1)
        s = best_t
        path.append(PAGES[s])
    return path, ret, s == GOAL


def simple_cycles(max_len: int = 6) -> list[list[int]]:
    """Every simple cycle in the page graph (self-loops included), as a list of pages, first page repeated last."""
    nbrs = {PAGES.index(s): {PAGES.index(t) for t in acts.values()} for s, acts in EDGES.items()}
    cycles = []
    for L in range(1, max_len + 1):
        for path in itertools.product(range(len(PAGES)), repeat=L):
            if path[0] != min(path) or len(set(path)) != L:      # canonical rotation, no repeated pages
                continue
            loop = list(path) + [path[0]]
            if all(b in nbrs[a] for a, b in zip(loop, loop[1:])):
                cycles.append(loop)
    return cycles


def cycle_gain(reward, cycle: list[int], gamma: float = 1.0) -> float:
    """Total (undiscounted by default) reward collected once around a cycle, with the sparse part removed."""
    return sum(reward(a, b, gamma) - sparse(a, b, gamma) for a, b in zip(cycle, cycle[1:]))


def q_learning(reward, gamma: float, horizon: int, episodes: int, seed: int = 0, lr: float = 0.5,
               explore: float = 0.2) -> tuple[float, float]:
    """Tabular epsilon-greedy Q-learning. Returns (greedy success rate, mean greedy steps) after training.

    The learned agent finds whatever the reward actually pays for, not what the reward's author meant.
    """
    rng = np.random.default_rng(seed)
    acts = {s: list(EDGES[PAGES[s]].items()) for s in range(len(PAGES))}
    Q = {s: np.zeros(len(acts[s])) for s in range(len(PAGES))}
    for _ in range(episodes):
        s = 0
        for step in range(horizon):
            i = rng.integers(len(acts[s])) if rng.random() < explore else int(np.argmax(Q[s]))
            t = PAGES.index(acts[s][i][1])
            last = step == horizon - 1
            target = reward(s, t, gamma, last) + (0.0 if t == GOAL or last else gamma * Q[t].max())
            Q[s][i] += lr * (target - Q[s][i])
            s = t
            if s == GOAL:
                break
    s, steps = 0, 0
    for steps in range(1, horizon + 1):
        s = PAGES.index(acts[s][int(np.argmax(Q[s]))][1])
        if s == GOAL:
            return 1.0, float(steps)
    return 0.0, float(horizon)
