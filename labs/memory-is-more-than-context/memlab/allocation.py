"""Section 4, "The allocation problem": pick context items under a token budget.

If utility were additive this would be a 0/1 knapsack. It is not: items can be duplicates,
complements or contradictions. The post's Figure 3 uses nine hand-set items. They are copied
here exactly, so the script can check what the figure claims and also find the true optimum.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    name: str
    u: float  # utility on its own
    tokens: int


@dataclass
class Pool:
    """Candidate items plus pairwise interactions, added to U when both items are chosen."""

    items: list[Item]
    pair_bonus: dict[tuple[int, int], float] = field(default_factory=dict)

    def utility(self, chosen) -> float:
        s = set(chosen)
        total = sum(self.items[i].u for i in s)
        for (i, j), bonus in self.pair_bonus.items():
            if i in s and j in s:
                total += bonus
        return total

    def tokens(self, chosen) -> int:
        return sum(self.items[i].tokens for i in chosen)


def figure3_pool(consistent: bool = False) -> Pool:
    """The nine items and three interactions from the post's Figure 3 (figures.js).

    The figure penalises the stale Jenkins fact (#3) only next to item #1. Item #2 states the
    same fact, so it contradicts #3 too. `consistent=True` adds that missing penalty.
    """
    items = [
        Item("user prefers dark mode", 0.06, 20),
        Item("project-x deploys via GitHub Actions", 0.85, 30),
        Item("deploys are done through GH Actions (restated)", 0.84, 34),
        Item("project-x deploys via Jenkins [2025, stale]", 0.70, 32),
        Item("deploy runbook section 3, full text", 0.95, 2200),
        Item("last deploy failed: missing scope read:user", 0.82, 42),
        Item("fix: add read:user to the staging app config", 0.68, 38),
        Item("user Q2 travel plans", -0.15, 260),
        Item("staging runners are on image v3", 0.45, 28),
    ]
    bonus = {
        (1, 2): -1.02 * min(0.85, 0.84),  # duplicate: the second copy adds (slightly less than) nothing
        (1, 3): -0.95,  # contradiction: current fact plus stale fact
        (5, 6): +0.28,  # complement: the failure plus its fix
    }
    if consistent:
        bonus[(2, 3)] = -0.95
    return Pool(items, bonus)


def select_static(pool: Pool, budget: int, key) -> list[int]:
    """Sort once by `key`, then add every item that still fits. Top-k relevance and density both work this way."""
    chosen: list[int] = []
    for i in sorted(range(len(pool.items)), key=key, reverse=True):
        if pool.tokens(chosen) + pool.items[i].tokens <= budget:
            chosen.append(i)
    return chosen


def select_relevance(pool: Pool, budget: int) -> list[int]:
    return select_static(pool, budget, key=lambda i: pool.items[i].u)


def select_density(pool: Pool, budget: int) -> list[int]:
    return select_static(pool, budget, key=lambda i: pool.items[i].u / pool.items[i].tokens)


def select_marginal(pool: Pool, budget: int) -> list[int]:
    """Greedy on marginal gain per token, Delta(c | C) / tokens(c). Stops when no item has positive gain."""
    chosen: list[int] = []
    while True:
        best, best_ratio = None, 0.0
        base = pool.utility(chosen)
        for i in range(len(pool.items)):
            if i in chosen or pool.tokens(chosen) + pool.items[i].tokens > budget:
                continue
            gain = pool.utility(chosen + [i]) - base
            if gain > 1e-9 and gain / pool.items[i].tokens > best_ratio:
                best, best_ratio = i, gain / pool.items[i].tokens
        if best is None:
            return chosen
        chosen.append(best)


def select_optimal(pool: Pool, budget: int) -> list[int]:
    """Exact answer by enumerating every subset. Only for small pools (2^n subsets)."""
    n = len(pool.items)
    best, best_u = [], 0.0
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            if pool.tokens(combo) <= budget:
                u = pool.utility(combo)
                if u > best_u + 1e-12:
                    best, best_u = list(combo), u
    return best


SELECTORS = {
    "top-k relevance": select_relevance,
    "density u/l": select_density,
    "marginal gain": select_marginal,
    "exact optimum": select_optimal,
}


def random_pool(rng: random.Random, n: int = 12) -> Pool:
    """A random pool with the same kinds of interaction as Figure 3, for a many-instance comparison."""
    items = []
    for k in range(n):
        tokens = int(rng.choice([rng.randint(15, 60), rng.randint(15, 60), rng.randint(200, 2500)]))
        u = rng.uniform(-0.2, 1.0)
        items.append(Item(f"item{k}", round(u, 3), tokens))
    bonus: dict[tuple[int, int], float] = {}
    pairs = list(itertools.combinations(range(n), 2))
    rng.shuffle(pairs)
    for (i, j) in pairs[:3]:  # duplicates
        bonus[(i, j)] = -min(max(items[i].u, 0), max(items[j].u, 0))
    for (i, j) in pairs[3:5]:  # contradictions
        bonus[(i, j)] = -rng.uniform(0.6, 1.2)
    for (i, j) in pairs[5:7]:  # complements
        bonus[(i, j)] = rng.uniform(0.1, 0.4)
    return Pool(items, bonus)
