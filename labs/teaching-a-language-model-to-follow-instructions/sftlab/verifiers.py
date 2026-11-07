"""Checkers for the 'can correctness be checked?' column of the taxonomy, plus data-construction helpers
(capped-proportional mixing, n-gram contamination, ROUGE-L)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile

from .metrics import words


def check_json(reply: str, schema: dict[str, type]) -> bool:
    """Structured output: parses, has exactly these keys, with these types."""
    try:
        obj = json.loads(reply)
    except Exception:
        return False
    return isinstance(obj, dict) and set(obj) == set(schema) and all(isinstance(obj[k], t) and not isinstance(obj[k], bool) for k, t in schema.items())


def check_label(reply: str, labels: set[str], gold: str) -> bool:
    """Classification: output only the label."""
    r = reply.strip().lower()
    return r in labels and r == gold


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(s: str):
    m = NUM.findall(s.replace("$", ""))
    return float(m[-1].replace(",", "")) if m else None


def check_math(reply: str, gold: float) -> bool:
    v = last_number(reply)
    return v is not None and abs(v - gold) < 1e-6


def check_code(code: str, tests: str, timeout: float = 5.0) -> bool:
    """Run the candidate and its asserts in a fresh interpreter with a time limit. Not a security sandbox:
    only run code you generated yourself in a throwaway environment."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + tests + "\n")
        path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def capped_proportional(sizes: dict[str, int], cap: int) -> dict[str, float]:
    """FLAN-style mixing: each source contributes min(size, cap) examples; weights are those counts, normalised."""
    c = {k: min(v, cap) for k, v in sizes.items()}
    t = sum(c.values())
    return {k: v / t for k, v in c.items()}


def proportional(sizes: dict[str, int]) -> dict[str, float]:
    t = sum(sizes.values())
    return {k: v / t for k, v in sizes.items()}


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    l = lcs_len(wa, wb)
    if l == 0:
        return 0.0
    p, r = l / len(wb), l / len(wa)
    return 2 * p * r / (p + r)


def contaminated_fraction(eval_texts: list[str], train_texts: list[str], n: int) -> tuple[float, list[int]]:
    """Share of eval texts that share at least one word n-gram with any training text."""
    from .metrics import ngrams
    train = set()
    for t in train_texts:
        train |= ngrams(t, n)
    hit = [i for i, e in enumerate(eval_texts) if ngrams(e, n) & train]
    return len(hit) / max(1, len(eval_texts)), hit
