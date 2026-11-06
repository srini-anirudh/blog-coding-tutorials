"""Observation policies for outputs that are too big to show the model.

A test run can print 50,000 lines. Pasting that into the context costs money,
pushes the task out of view, and buries the one line that matters. The policy:

1. keep the head (how the command started) and the tail (how it ended),
2. rescue error-looking lines from the middle, with a line of context,
3. clip absurdly long lines (minified files, progress bars),
4. say exactly what was dropped, and where the full text lives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ERROR_PATTERNS = re.compile(
    r"(Traceback|Error\b|error:|FAILED|FAIL:|assert|Exception|panic|fatal|"
    r"E\s{3}|\bfailed\b|^>\s)", re.I)


@dataclass
class Truncated:
    text: str
    total_lines: int
    shown_lines: int
    rescued_lines: int
    truncated: bool


def head_only(text: str, max_lines: int) -> str:
    """The naive policy, kept for comparison: what most quick harnesses do."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} more lines]"


def truncate_output(text: str, max_lines: int = 120, max_line_chars: int = 400,
                    head: int = 30, tail: int = 50, max_rescued: int = 30,
                    spill_path: Path | None = None, spill_name: str | None = None) -> Truncated:
    """Head + tail + rescued error lines. `max_lines` bounds the shown line count.

    If `spill_path` is given, the full text is written there and the footer tells the
    model to read `spill_name` (the workspace-relative path) for the rest.
    """
    lines = [ln if len(ln) <= max_line_chars else ln[:max_line_chars] + f" ...[{len(ln) - max_line_chars} chars]"
             for ln in text.splitlines()]
    n = len(lines)
    if n <= max_lines:
        return Truncated("\n".join(lines), n, n, 0, truncated=False)

    head, tail = min(head, max_lines // 3), min(tail, max_lines // 2)
    keep = set(range(head)) | set(range(n - tail, n))
    rescued = 0
    budget = min(max_rescued, max_lines - len(keep))
    for i in range(head, n - tail):
        if rescued >= budget:
            break
        if ERROR_PATTERNS.search(lines[i]):
            for j in (i - 1, i, i + 1):
                if head <= j < n - tail and j not in keep and rescued < budget:
                    keep.add(j)
                    rescued += 1

    out: list[str] = []
    prev = -1
    for i in sorted(keep):
        if i != prev + 1:
            out.append(f"... [{i - prev - 1} lines omitted] ...")
        out.append(lines[i])
        prev = i
    if spill_path is not None:
        spill_path.parent.mkdir(parents=True, exist_ok=True)
        spill_path.write_text(text)
        out.append(f"[output truncated: {n} lines total; full text saved to {spill_name or spill_path.name}, "
                   f"read it with read_file if you need more]")
    else:
        out.append(f"[output truncated: showing {len(keep)} of {n} lines]")
    return Truncated("\n".join(out), n, len(keep), rescued, truncated=True)
