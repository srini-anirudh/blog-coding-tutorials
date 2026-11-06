"""Self-consistency as an estimator (section "Vote").

A single chain is one sample from the answer marginal q(y). Plurality voting over N chains converges to
argmax_y q(y), so it recovers the right answer exactly when the right answer is the mode of q, not when
q(correct) > 1/2. Best-of-N with a perfect verifier (pass@N) is the ceiling any selector can reach.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Hashable, Sequence


def plurality(answers: Sequence[Hashable | None]) -> Hashable | None:
    """Most common non-None answer. Ties go to the answer that reached the top count first."""
    counts: Counter = Counter()
    best, best_count = None, 0
    for a in answers:
        if a is None:
            continue
        counts[a] += 1
        if counts[a] > best_count:
            best, best_count = a, counts[a]
    return best


def simulate_vote(q: dict[Hashable, float], correct: Hashable, n: int, trials: int = 20_000, seed: int = 0) -> float:
    """Monte Carlo P(plurality of n draws from q == correct). Mass not in q means 'no answer'."""
    rng = random.Random(seed)
    keys = list(q) + [None]
    weights = list(q.values()) + [max(0.0, 1.0 - sum(q.values()))]
    hits = 0
    for _ in range(trials):
        draws = rng.choices(keys, weights=weights, k=n)
        hits += plurality(draws) == correct
    return hits / trials


def maj_at_k(answers: Sequence[Hashable | None], correct: Hashable, k: int, resamples: int = 64, seed: int = 0) -> float:
    """Expected accuracy of a k-sample vote, estimated by drawing k of the n samples without replacement."""
    n = len(answers)
    if k >= n:
        return float(plurality(answers) == correct)
    rng = random.Random(seed)
    return sum(plurality(rng.sample(list(answers), k)) == correct for _ in range(resamples)) / resamples


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased P(at least one of k samples is correct) from c correct out of n (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def answer_marginal(answers: Sequence[Hashable | None]) -> list[tuple[Hashable | None, float]]:
    """Empirical q(y), most common first."""
    n = len(answers)
    return [(a, c / n) for a, c in Counter(answers).most_common()]
