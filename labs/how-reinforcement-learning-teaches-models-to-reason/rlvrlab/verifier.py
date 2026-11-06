"""Section 1-2 of the post: a reward you write down, a reward you learn, and best-of-N against each.

`math_reward` and `format_reward` are the two R1-Zero style rule rewards. `surface_proxy` is a stand-in for a
learned preference model: it scores what a fluent answer looks like and never checks the arithmetic. It is written
by hand, so it is a caricature of a reward model, not a trained one.

`best_of_n` simulates reranking a pool of N sampled solutions with an oracle checker and with an imperfect scorer
whose errors have a heavy tail. It is a simulation with chosen parameters, not a model run.
"""

from __future__ import annotations

import re

import numpy as np

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
ANSWER_LINE = re.compile(r"answer is\s*(-?[\d,]+(?:\.\d+)?)", re.I)
THINK = re.compile(r"\s*<think>(.+?)</think>(.*)", re.S)


def extract_answer(text: str) -> str | None:
    """Last \\boxed{...}, else the last 'answer is N'. None when the response states no final answer."""
    boxed = BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    line = ANSWER_LINE.findall(text)
    return line[-1] if line else None


def _normalize(ans: str) -> str:
    ans = ans.replace(",", "").replace(" ", "").rstrip(".")
    try:
        x = float(ans)
        return str(int(x)) if x == int(x) else str(x)
    except ValueError:
        return ans


def math_reward(text: str, truth: str) -> float:
    """r(x, y) = 1[ extract_answer(y) == y* ]."""
    ans = extract_answer(text)
    return float(ans is not None and _normalize(ans) == _normalize(truth))


def format_reward(text: str) -> float:
    """R1-Zero's format reward: reasoning inside <think>...</think>, then something after it."""
    m = THINK.fullmatch(text)
    return float(bool(m and m.group(1).strip() and m.group(2).strip()))


CONFIDENT = ("clearly", "therefore", "first", "then", "finally", "so the answer")


def surface_proxy(text: str) -> float:
    """A hand-written caricature of a learned reward: rewards length, structure and confident words, in [0, 1]."""
    words = text.lower().split()
    length = min(len(words) / 60.0, 1.0)
    confident = min(sum(text.lower().count(w) for w in CONFIDENT) / 4.0, 1.0)
    structure = min(text.count("\n") / 4.0, 1.0)
    return round(0.4 * length + 0.4 * confident + 0.2 * structure, 3)


def best_of_n(
    ns: list[int],
    n_problems: int = 2000,
    signal: float = 2.0,
    noise: float = 1.0,
    fool_rate: float = 0.004,
    fool_boost: float = 4.0,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Accuracy of picking the top-scored of N candidates, for each N in `ns`.

    Each problem has a solve rate p ~ U(0.02, 0.5). A candidate is correct with probability p.
    Imperfect scorer: score = signal * correct + noise * N(0, 1), and a wrong candidate is, with probability
    `fool_rate`, one the scorer is confidently wrong about (+fool_boost). The oracle sees correctness exactly,
    so its best-of-N accuracy is pass@N.
    """
    rng = np.random.default_rng(seed)
    nmax = max(ns)
    p = rng.uniform(0.02, 0.5, size=(n_problems, 1))
    correct = rng.random((n_problems, nmax)) < p
    fool = (~correct) & (rng.random((n_problems, nmax)) < fool_rate)
    score = signal * correct + noise * rng.standard_normal((n_problems, nmax)) + fool_boost * fool
    out: dict[str, list[float]] = {"n": list(ns), "scorer": [], "oracle": [], "first_sample": []}
    for n in ns:
        pick = score[:, :n].argmax(axis=1)
        out["scorer"].append(float(correct[np.arange(n_problems), pick].mean()))
        out["oracle"].append(float(correct[:, :n].any(axis=1).mean()))
        out["first_sample"].append(float(correct[:, 0].mean()))
    return out
