"""Figure 3: fill a context budget three ways, then compare with the exact optimum.

    python labs/memory-is-more-than-context/scripts/01_allocation.py
"""

from __future__ import annotations

import random
import statistics

from _common import Report

from memlab.allocation import SELECTORS, figure3_pool, random_pool

out = Report("01_allocation")
data: dict = {"figure3": {}, "random_pools": {}}

for consistent in (False, True):
    pool = figure3_pool(consistent)
    label = "Figure 3 pool, with the missing #2-#3 contradiction added" if consistent else "Figure 3 pool, as in the post"
    out(f"\n{label}")
    for budget in (300, 2300, 2400):
        out(f"  budget B={budget}")
        for name, select in SELECTORS.items():
            chosen = sorted(select(pool, budget))
            u = pool.utility(chosen)
            out(f"    {name:<16} U={u:5.2f}  tokens={pool.tokens(chosen):5d}  items={chosen}")
            data["figure3"][f"{'consistent' if consistent else 'post'}/B={budget}/{name}"] = {
                "U": round(u, 4), "tokens": pool.tokens(chosen), "items": chosen}

out("\nItems: " + "; ".join(f"#{i} {it.name} (u={it.u}, {it.tokens} tok)" for i, it in enumerate(figure3_pool().items)))

rng = random.Random(0)
ratios: dict[str, list[float]] = {n: [] for n in SELECTORS}
n_pools = 500
for _ in range(n_pools):
    pool = random_pool(rng)
    budget = rng.choice([150, 300, 600, 2500])
    best = pool.utility(SELECTORS["exact optimum"](pool, budget))
    for name, select in SELECTORS.items():
        ratios[name].append(pool.utility(select(pool, budget)) / best if best > 0 else 1.0)

out(f"\n{n_pools} random pools of 12 items (3 duplicate pairs, 2 contradictions, 2 complements), "
    "utility as a share of the exact optimum")
out(f"  {'rule':<16}{'mean':>7}{'worst':>8}{'optimal':>9}")
for name, r in ratios.items():
    hits = sum(x > 1 - 1e-9 for x in r)
    out(f"  {name:<16}{statistics.mean(r):7.3f}{min(r):8.3f}{hits:>5d}/{n_pools}")
    data["random_pools"][name] = {"mean": round(statistics.mean(r), 4), "worst": round(min(r), 4), "optimal": hits}
out.save(data)
