"""Small text metrics: token F1 against a reference, distinct-n, edit-free near-duplicate check."""

from __future__ import annotations

import re
from collections import Counter

WORD = re.compile(r"\w+")


def words(s: str) -> list[str]:
    return WORD.findall(s.lower())


def f1(pred: str, ref: str) -> float:
    p, r = Counter(words(pred)), Counter(words(ref))
    common = sum((p & r).values())
    if not common:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def distinct_n(texts: list[str], n: int = 2) -> float:
    grams = []
    for t in texts:
        w = words(t)
        grams += [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def ngrams(s: str, n: int) -> set:
    w = words(s)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
