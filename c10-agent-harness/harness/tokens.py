"""A cheap, deterministic token estimate for budgeting context.

Real tokenizers differ by model, and calling one on every context build is slow.
For budgets we only need a stable estimate. Splitting into words and single
punctuation marks lands close to BPE counts on English and code, and overcounts
JSON (scripts/show_schemas.py measures all three against a real tokenizer). The HF
client reports exact counts in its usage numbers, so the trace shows both.
"""

from __future__ import annotations

import re

_PIECES = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Estimated token count. Long words count as several tokens, as in BPE."""
    return sum(1 + len(p) // 8 for p in _PIECES.findall(text))


def count_message_tokens(messages: list[dict]) -> int:
    """Tokens for a chat: content, tool-call arguments and ~4 tokens of framing per message."""
    total = 0
    for m in messages:
        total += 4 + count_tokens(m.get("content") or "")
        for c in m.get("tool_calls", []):
            total += count_tokens(c["function"]["name"]) + count_tokens(str(c["function"]["arguments"]))
    return total
