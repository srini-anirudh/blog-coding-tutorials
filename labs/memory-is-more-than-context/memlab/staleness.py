"""Section 11, "Forgetting is a feature": the 1/(lambda t) law and precision at fixed k.

The closed forms are the post's. The simulations draw real random facts with exponential
lifetimes, so the two can be compared. Units are "steps", as in Figure 10.
"""

from __future__ import annotations

import math

import torch


def rho_append(t: float, lam: float) -> float:
    """Fraction of an append-only store that is still true: (1 - exp(-lam t)) / (lam t)."""
    if t <= 0:
        return 1.0
    return (1.0 - math.exp(-lam * t)) / (lam * t)


def rho_managed(t: float, lam: float, recall: float) -> float:
    """Figure 10's managed store: a deletion pass removes a fraction `recall` of the stale records."""
    if t <= 0:
        return 1.0
    true = (1.0 - math.exp(-lam * t)) / lam
    stale = t - true
    return true / (true + (1.0 - recall) * stale)


def simulate_store(steps: int, arrivals: int, lam: float, recall: float = 0.0, seed: int = 0) -> list[dict]:
    """Add `arrivals` facts per step; each stays true for an Exp(lam) time.

    When a fact goes stale, the deletion policy notices it with probability `recall` and removes
    it. A fact it misses stays in the store for good. recall=0 is the append-only store.
    Returns, for each step, the record count and the fraction still true.
    """
    g = torch.Generator().manual_seed(seed)
    n = steps * arrivals
    born = torch.arange(steps).repeat_interleave(arrivals).double()
    # Arrivals are spread over the step, so the discrete count matches the continuous a*t.
    born = born + torch.rand(n, generator=g, dtype=torch.float64)
    dies = born - torch.log(torch.rand(n, generator=g, dtype=torch.float64)) / lam
    caught = torch.rand(n, generator=g, dtype=torch.float64) < recall
    rows = []
    for t in range(1, steps + 1):
        present = born < t
        true = present & (dies > t)
        kept = present & (true | ~caught)
        rows.append({"t": t, "records": int(kept.sum()), "true": int(true.sum()),
                     "rho": float(true.sum()) / max(int(kept.sum()), 1)})
    return rows


def precision_at_k(n: int, n_relevant: int, k: int, mu_rel: float, mu_irr: float, sigma: float,
                   trials: int = 200, seed: int = 0) -> float:
    """Mean precision@k when relevant scores are N(mu_rel, sigma^2) and the n - n_relevant others N(mu_irr, sigma^2)."""
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(trials):
        rel = mu_rel + sigma * torch.randn(n_relevant, generator=g)
        irr = mu_irr + sigma * torch.randn(n - n_relevant, generator=g)
        scores = torch.cat([rel, irr])
        top = torch.topk(scores, k).indices
        total += float((top < n_relevant).sum()) / k
    return total / trials
