"""Selection: turning a pile of samples into one answer.

Three selectors: majority vote, best-of-N under a score, and a verifier filter that picks
uniformly among the samples that pass. Plus the false-positive ceiling p / (p + (1 - p) phi).
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def majority_vote(answers, rng: np.random.Generator):
    """Most common answer. Ties are broken at random. None (unparsed) never wins unless alone."""
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None
    top = max(counts.values())
    tied = [a for a, c in counts.items() if c == top]
    return tied[rng.integers(len(tied))]


def modal_answer_correct(answers, truth) -> bool:
    """Is the most common answer the correct one? The large-k limit of majority voting."""
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return False
    top = max(counts.values())
    return counts.get(truth, 0) == top and sum(c == top for c in counts.values()) == 1


def fp_ceiling(p: float, phi: float) -> float:
    """Accuracy limit of 'resample, keep a passer' when wrong samples pass with rate phi."""
    return p / (p + (1.0 - p) * phi)


def verifier_accuracy(p: float, phi: float, k) -> np.ndarray:
    """Exact accuracy after k samples when you pick uniformly among passers.

    Each sample passes with r = p + (1 - p) phi. A passer is correct with chance p / r.
    So accuracy = P(at least one passer) * p / r. As k grows it tends to p / r.
    """
    r = p + (1.0 - p) * phi
    if r == 0:
        return np.zeros_like(np.asarray(k, float))
    return (1.0 - (1.0 - r) ** np.asarray(k, float)) * p / r


def verifier_accuracy_mc(p: float, phi: float, k: int, trials: int, rng: np.random.Generator) -> float:
    """Monte Carlo check of verifier_accuracy."""
    n_c = rng.binomial(k, p, size=trials)
    n_w = rng.binomial(k - n_c, phi)
    passers = n_c + n_w
    acc = np.where(passers > 0, n_c / np.maximum(passers, 1), 0.0)
    return float(acc.mean())


def selector_curves(problems, ks, n_draws: int, rng: np.random.Generator) -> dict:
    """Accuracy of each selector on subsets of k samples, averaged over problems and random subsets.

    problems: list of dicts with 'truth', 'answers' (list), 'scores' (list of float),
    and optionally 'passes' (list of bool, an imperfect verifier).
    """
    from .coverage import pass_at_k

    out = {name: [] for name in ("coverage", "majority", "best_score", "verifier")}
    for k in ks:
        acc = {name: [] for name in out}
        for pr in problems:
            ans, n = pr["answers"], len(pr["answers"])
            correct = np.array([a == pr["truth"] for a in ans])
            acc["coverage"].append(pass_at_k(n, int(correct.sum()), k))
            mv = bs = vf = 0.0
            for _ in range(n_draws):
                idx = rng.choice(n, size=k, replace=False)
                mv += majority_vote([ans[i] for i in idx], rng) == pr["truth"]
                scores = np.array([pr["scores"][i] for i in idx])
                bs += correct[idx[int(np.argmax(scores))]]
                if "passes" in pr:
                    ok = [i for i in idx if pr["passes"][i]]
                    vf += correct[ok[rng.integers(len(ok))]] if ok else 0.0
            acc["majority"].append(mv / n_draws)
            acc["best_score"].append(bs / n_draws)
            acc["verifier"].append(vf / n_draws)
        for name in out:
            out[name].append(float(np.mean(acc[name])))
    return out
