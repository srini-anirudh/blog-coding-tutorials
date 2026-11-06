"""Rule-based quality filters in the style of Gopher (Rae et al., 2021) and C4 (Raffel et al., 2019).

Each rule is a hand-written guess about what real prose looks like. `gopher_reasons` and
`c4_reasons` return the list of rules a document fails. An empty list means the document is kept.
The thresholds follow the published values where the papers state them.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

STOPWORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}  # Gopher's list
TERMINAL = ('.', '!', '?', '"', "'")
BULLETS = ("-", "*", "•", "‣", "◦")


def words(text: str) -> list[str]:
    return text.split()


def _dup_ngram_char_frac(ws: list[str], n: int) -> float:
    """Share of characters inside n-grams that occur more than once (Gopher's repetition rule)."""
    if len(ws) < n:
        return 0.0
    grams = [tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(ws)
    for i, g in enumerate(grams):
        if counts[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    total = sum(len(w) for w in ws)
    return sum(len(w) for w, c in zip(ws, covered) if c) / max(total, 1)


def _top_ngram_char_frac(ws: list[str], n: int) -> float:
    """Characters taken up by the single most frequent n-gram."""
    if len(ws) < n:
        return 0.0
    counts = Counter(tuple(ws[i : i + n]) for i in range(len(ws) - n + 1))
    gram, c = counts.most_common(1)[0]
    return c * sum(len(w) for w in gram) / max(sum(len(w) for w in ws), 1)


def gopher_reasons(text: str) -> list[str]:
    """Gopher's quality and repetition rules."""
    ws = words(text)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    out = []
    if not 50 <= len(ws) <= 100_000:
        out.append("word_count")
        if not ws:
            return out
    mean_len = sum(len(w) for w in ws) / len(ws)
    if not 3 <= mean_len <= 10:
        out.append("mean_word_length")
    if (text.count("#") + text.count("...") + text.count("…")) / len(ws) > 0.1:
        out.append("symbol_ratio")
    if lines and sum(ln.lstrip().startswith(BULLETS) for ln in lines) / len(lines) > 0.9:
        out.append("bullet_lines")
    if lines and sum(ln.rstrip().endswith(("...", "…")) for ln in lines) / len(lines) > 0.3:
        out.append("ellipsis_lines")
    if sum(bool(re.search(r"[A-Za-z]", w)) for w in ws) / len(ws) < 0.8:
        out.append("alpha_words")
    if sum(w.lower().strip(".,;:!?\"'()") in STOPWORDS for w in ws) < 2:
        out.append("stopwords")
    if lines and 1 - len(set(lines)) / len(lines) > 0.3:
        out.append("dup_lines")
    if _top_ngram_char_frac(ws, 2) > 0.20:
        out.append("top_2gram")
    if _dup_ngram_char_frac(ws, 5) > 0.15:
        out.append("dup_5gram")
    return out


def c4_line_filter(text: str) -> list[str]:
    """C4's line rules: keep lines that end in terminal punctuation and have at least 5 words."""
    kept = []
    for ln in text.split("\n"):
        s = ln.strip()
        if "javascript" in s.lower():
            continue
        if s.endswith(TERMINAL) and len(s.split()) >= 5:
            kept.append(s)
    return kept


def c4_reasons(text: str) -> list[str]:
    """C4's page rules, applied after the line filter."""
    out = []
    if "lorem ipsum" in text.lower():
        out.append("lorem_ipsum")
    if "{" in text:
        out.append("curly_bracket")
    if len(c4_line_filter(text)) < 3:
        out.append("too_few_sentences")
    return out


def quality_reasons(text: str) -> list[str]:
    return gopher_reasons(text) + c4_reasons(text)


def exact_hash(text: str) -> str:
    """Exact-duplicate key: a hash of the whitespace-normalised document."""
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()


def ngram_set(text: str, n: int = 13) -> set[tuple[str, ...]]:
    ws = [w.lower() for w in re.findall(r"\w+", text)]
    return {tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)}


def contaminated(text: str, eval_ngrams: set[tuple[str, ...]], n: int = 13) -> bool:
    """True if the document shares any n-gram with the evaluation set (13-gram overlap, as in GPT-3)."""
    return not ngram_set(text, n).isdisjoint(eval_ngrams)
