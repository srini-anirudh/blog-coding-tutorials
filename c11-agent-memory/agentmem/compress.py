"""Context compression: four ways to make text cheaper before it is dropped.

    extraction     keep the lines that carry the signal (errors, facts), drop the rest
    summarisation  replace old turns with a shorter account of them (extractive or by a model)
    dropping       remove whole blocks by priority (that is the budgeter's job)
    ordering       put the most valuable blocks where the model reads best: the edges

Each strategy loses information in a different way. Extraction loses context around
the signal. Summarisation loses detail and can lose it silently. Dropping loses whole
items. Ordering loses nothing, which is why it is applied last and always.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence, TypeVar

from .budget import Block
from .semantic import RuleFactExtractor
from .tokens import count_tokens, truncate_to_tokens

T = TypeVar("T")

SIGNAL_RE = re.compile(r"(error|exception|fail|failed|assert|traceback|warning|exit code [1-9])", re.I)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def extract_lines(text: str, max_tokens: int, pattern: re.Pattern[str] = SIGNAL_RE, context: int = 1,
                  counter: Callable[[str], int] = count_tokens) -> str:
    """Keep matching lines plus `context` lines around each; mark the gaps.

    For a 2,000-line test log the failure is a dozen lines. Head-and-tail truncation
    keeps the first and last lines and may cut exactly the dozen that matter.
    """
    lines = text.splitlines()
    keep = set()
    for i, line in enumerate(lines):
        if pattern.search(line):
            keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))
    if not keep:
        return truncate_to_tokens(text, max_tokens, counter, keep="both")
    out, prev = [], -1
    for i in sorted(keep):
        if i != prev + 1:
            out.append(f"[... {i - prev - 1} lines ...]")
        out.append(lines[i])
        prev = i
    if prev < len(lines) - 1:
        out.append(f"[... {len(lines) - 1 - prev} lines ...]")
    joined = "\n".join(out)
    if context > 0 and counter(joined) > max_tokens:
        return extract_lines(text, max_tokens, pattern, 0, counter)  # the signal outranks its context
    return truncate_to_tokens(joined, max_tokens, counter, keep="both")


def extractive_summary(turns: Sequence[str], max_tokens: int, counter: Callable[[str], int] = count_tokens) -> str:
    """Summarise old turns without a model: extracted facts first, then first sentences.

    Facts are the part of a conversation most likely to be needed later, so they are
    kept verbatim; everything else is cut to its first sentence.
    """
    extract = RuleFactExtractor()
    facts: dict[tuple[str, str], str] = {}
    other: list[str] = []
    for turn in turns:
        claims = extract(turn, t=0.0)
        for c in claims:
            if c.kind == "delete":
                facts.pop((c.subject, c.attribute), None)
            elif c.scope != "session" and c.confidence >= 0.5:
                facts[(c.subject, c.attribute)] = f"{c.subject} {c.attribute}: {c.value}"  # later wins
        if not claims and turn.strip():  # a turn with a claim must not bring its stale wording back
            other.append(_SENTENCE_RE.split(turn.strip())[0])
    lines = [f"- {f}" for f in facts.values()] + [f"- {o}" for o in other]
    out, used = [], 0
    for line in lines:
        cost = counter(line) + 1
        if used + cost > max_tokens:
            break
        out.append(line)
        used += cost
    return "\n".join(out)


SUMMARY_PROMPT = """Summarise the conversation below for an assistant that will continue it.
Keep every fact a later task could need (names, values, commands, decisions). Drop chit-chat.
At most {max_words} words.

{text}

Summary:"""


def model_summary(turns: Sequence[str], max_tokens: int, complete: Callable[[str], str],
                  counter: Callable[[str], int] = count_tokens) -> str:
    """Abstractive summary by any text-completion callable, hard-capped at max_tokens.

    The cap is enforced after the call because models do not reliably obey length
    instructions.
    """
    raw = complete(SUMMARY_PROMPT.format(max_words=max(10, int(max_tokens * 0.7)), text="\n".join(turns)))
    return truncate_to_tokens(raw.strip(), max_tokens, counter)


def compact_history(events: list[Block], keep_recent: int, max_summary_tokens: int,
                    summarize: Callable[[Sequence[str], int], str] = extractive_summary) -> list[Block]:
    """Replace all but the newest `keep_recent` events with one summary block."""
    events = sorted(events, key=lambda b: b.order)
    split = max(0, len(events) - keep_recent)
    old, recent = events[:split], events[split:]
    if not old:
        return recent
    summary = summarize([b.text for b in old], max_summary_tokens)
    head = Block("events", f"(summary of {len(old)} earlier events)\n{summary}",
                 value=max(b.value for b in old), order=old[0].order, id="summary")
    return [head] + recent


def edge_order(items: Sequence[T], key: Callable[[T], float]) -> list[T]:
    """Place the best item first, the second best last, the third second, and so on.

    Long-context models use the start and the end of a prompt more reliably than the
    middle (Liu et al., 2023, "Lost in the Middle"). This puts the weakest items there.
    """
    ranked = sorted(items, key=key, reverse=True)
    front, back = ranked[0::2], ranked[1::2]
    return front + back[::-1]
