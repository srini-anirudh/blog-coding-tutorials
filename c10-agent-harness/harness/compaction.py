"""Context compaction: when the history outgrows its budget, summarise the old turns.

Rules, in order of priority:

1. Task state is never in the history (the context builder re-renders the task and
   the plan every turn), so compaction cannot lose it.
2. Important failures from the dropped turns are kept verbatim (first lines only):
   a failing assertion is the most expensive fact to rediscover.
3. Older turns become one line each: which tool, which arguments, what happened.
4. The most recent turns are kept whole, and a cut never separates a tool call
   from its result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .messages import Message

Summarizer = Callable[[list[Message]], str]
SUMMARY_HEADER = "[Summary of earlier turns, compacted by the harness]"


@dataclass
class CompactionResult:
    cut: int                 # index into the history: messages[cut:] are kept whole
    summary: str
    dropped: int


def turn_starts(history: list[Message]) -> list[int]:
    """Indices of assistant messages: the only safe places to cut."""
    return [i for i, m in enumerate(history) if m.role == "assistant"]


def _short_args(args: dict, width: int = 60) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return text if len(text) <= width else text[: width - 3] + "..."


def _is_failure(text: str) -> bool:
    return text.startswith(("Error", "Permission denied", "Tool call")) or "FAIL" in text[:200]


def _split_summary(text: str) -> tuple[list[str], list[str], list[str]]:
    """Undo `extractive_summary` so a second compaction extends the first instead of nesting it."""
    lines, touched, failures, in_failures = [], [], [], False
    for line in text.splitlines()[1:]:
        if line.startswith("Files edited so far: "):
            touched += line.removeprefix("Files edited so far: ").split(", ")
        elif line == "Important failures (verbatim):":
            in_failures = True
        elif in_failures:
            failures.append(line[2:])
        else:
            lines.append(line)
    return lines, touched, failures


def extractive_summary(messages: list[Message], max_failure_lines: int = 24) -> str:
    """A deterministic summary: one line per tool call, plus verbatim failure heads."""
    results = {m.tool_call_id: m for m in messages if m.role == "tool"}
    lines: list[str] = []
    touched: list[str] = []
    failures: list[str] = []
    for m in messages:
        if m.role == "user" and m.content.startswith(SUMMARY_HEADER):
            old_lines, old_touched, old_failures = _split_summary(m.content)
            lines, touched, failures = lines + old_lines, touched + old_touched, failures + old_failures
        elif m.role == "user" and m.content:
            lines.append(f"- user: {m.content.splitlines()[0][:100]}")
        for c in m.tool_calls:
            r = results.get(c.id)
            head = r.content.strip().splitlines()[0][:100] if r and r.content.strip() else "(no result)"
            lines.append(f"- {c.name}({_short_args(c.arguments)}) -> {head}")
            if c.name in ("edit_file", "write_file", "apply_patch") and "path" in c.arguments:
                touched.append(c.arguments["path"])
            if r is not None and _is_failure(r.content):
                failures += r.content.strip().splitlines()[:6]
    out = [SUMMARY_HEADER] + lines
    if touched:
        out.append("Files edited so far: " + ", ".join(dict.fromkeys(touched)))
    if failures:
        # Most recent failures matter most, so a long list keeps its tail.
        out.append("Important failures (verbatim):")
        out += [f"  {line}" for line in failures[-max_failure_lines:]]
    return "\n".join(out)


def plan_compaction(history: list[Message], keep_recent_turns: int = 4,
                    summarizer: Summarizer = extractive_summary) -> CompactionResult | None:
    """Choose a cut that keeps the last `keep_recent_turns` turns, and summarise the rest."""
    starts = turn_starts(history)
    if len(starts) <= keep_recent_turns:
        return None  # nothing old enough to drop
    cut = starts[-keep_recent_turns]
    old = history[:cut]
    return CompactionResult(cut, summarizer(old), dropped=len(old))
