"""Token counting and token-aware truncation.

Every decision in this package is priced in tokens, so the counter is the unit of
account. The default counter is a regex word-piece estimate: deterministic, fast and
dependency-free. `HFTokenCounter` gives the exact count for a real tokenizer.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Literal, Protocol

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class TokenCounter(Protocol):
    def __call__(self, text: str) -> int: ...


@lru_cache(maxsize=65536)
def count_tokens(text: str) -> int:
    """Estimated token count: one per word or punctuation mark, more for long words.

    This is the same estimate the C10 harness budgets with, so a budget means the
    same thing in both packages. It lands near a BPE count on English and code (the
    README reports the measured ratio against the SmolLM2 tokenizer).
    """
    return sum(1 + len(p) // 8 for p in _TOKEN_RE.findall(text))


class HFTokenCounter:
    """Exact counts from a Hugging Face tokenizer, e.g. the model the agent runs."""

    def __init__(self, name: str = "HuggingFaceTB/SmolLM2-135M-Instruct") -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def __call__(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_to_tokens(text: str, max_tokens: int, counter: Callable[[str], int] = count_tokens,
                       keep: Literal["head", "tail", "both"] = "head") -> str:
    """Cut `text` to at most `max_tokens`, cutting on line boundaries where possible.

    keep="both" keeps the head and the tail and marks the gap. That is the right
    default for tool output: the command echo is at the top, the error at the bottom.
    """
    if counter(text) <= max_tokens:
        return text
    if max_tokens <= 0:
        return ""
    lines = text.splitlines()
    if keep == "both":
        marker_budget = 12  # "[... N lines omitted ...]" costs 11 tokens
        half = max(1, (max_tokens - marker_budget) // 2)
        head = _take_lines(lines, half, counter)
        tail = _take_lines(lines[::-1], half, counter)[::-1]
        if not head and not tail:
            return _cut_words(text, max_tokens, counter, "head")
        omitted = len(lines) - len(head) - len(tail)
        return "\n".join(head + [f"[... {omitted} lines omitted ...]"] + tail)
    if keep == "tail":
        kept = _take_lines(lines[::-1], max_tokens, counter)[::-1]
    else:
        kept = _take_lines(lines, max_tokens, counter)
    if kept:
        return "\n".join(kept)
    return _cut_words(text, max_tokens, counter, keep)


def _cut_words(text: str, max_tokens: int, counter: Callable[[str], int], keep: str) -> str:
    """One line longer than the budget: cut on words, so the result still respects the count."""
    words = text.split()
    if keep == "tail":
        words = words[::-1]
    out: list[str] = []
    for word in words:
        if counter(" ".join(out + [word])) > max_tokens:
            break
        out.append(word)
    return " ".join(out[::-1] if keep == "tail" else out)


def _take_lines(lines: list[str], max_tokens: int, counter: Callable[[str], int]) -> list[str]:
    out: list[str] = []
    used = 0
    for line in lines:
        cost = counter(line) + 1  # +1 for the newline that joins it
        if used + cost > max_tokens:
            break
        out.append(line)
        used += cost
    return out
