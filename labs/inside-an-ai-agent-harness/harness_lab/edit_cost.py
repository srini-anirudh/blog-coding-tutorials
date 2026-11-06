"""The arithmetic of a failed patch (post, section 8), with C10's real appliers.

An attempt in format F costs m_F output tokens and lands with probability p_F.
Retries are geometric, so one landed edit costs m_F / p_F on average. The token
costs m_F are measured on real files. The success probabilities p_F are parameters:
the post calls them schematic, and so do we.

A failed attempt is a real failure, not a coin flip. With probability 1 - p the
simulated model makes one slip, and the C10 applier (or a guard) refuses it:

    unified diff     stale line numbers  -> apply_hunks(strict=True) raises EditError
    search/replace   misquoted line      -> apply_search_replace raises EditError
    whole file       output cut short    -> a guard refuses a write that loses the file's tail

The error message is appended to the context, so it is paid again on every later step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from harness.editing import EditError, apply_hunks, apply_search_replace, make_diff, parse_unified_diff
from harness.tokens import count_tokens

FORMATS = ("whole file", "search/replace", "unified diff")


def target_edit(text: str, line: int) -> str:
    lines = text.splitlines()
    lines[line] = lines[line] + "  # edited"
    return "\n".join(lines) + "\n"


def payload(fmt: str, text: str, line: int, slip: bool) -> str:
    """What the model writes for a one-line change at `line`, with or without a slip."""
    new = target_edit(text, line)
    lines = text.splitlines()
    if fmt == "whole file":
        return new[: len(new) // 2] if slip else new
    if fmt == "search/replace":
        old = lines[max(0, line - 1): line + 2]
        rep = old[:]
        rep[min(1, line)] = rep[min(1, line)] + "  # edited"
        if slip:
            old = [old[0] + " x"] + old[1:]
        return "<<<SEARCH\n" + "\n".join(old) + "\n===\n" + "\n".join(rep) + "\n>>>REPLACE"
    if fmt == "unified diff":
        diff = make_diff(text, new, "f.py")
        if slip:  # the model is unsure which line it is on: the hunk header is off by 7
            head = [ln for ln in diff.splitlines() if ln.startswith("@@")][0]
            a = int(head.split()[1].split(",")[0][1:])
            diff = diff.replace(head, head.replace(f"-{a}", f"-{a + 7}", 1), 1)
        return diff
    raise ValueError(fmt)


def apply(fmt: str, text: str, body: str) -> str:
    """Apply through C10's appliers. Raises EditError on a refusal."""
    if fmt == "whole file":
        if len(body.splitlines()) < len(text.splitlines()):
            raise EditError(f"refused: the new file has {len(body.splitlines())} lines, the old one "
                            f"{len(text.splitlines())}. Write the whole file, or use a patch.")
        return body
    if fmt == "search/replace":
        search = body.split("<<<SEARCH\n", 1)[1].split("\n===\n", 1)[0]
        replace = body.split("\n===\n", 1)[1].rsplit("\n>>>REPLACE", 1)[0]
        return apply_search_replace(text, search, replace).text
    patch = parse_unified_diff(body)[0]
    return apply_hunks(text, patch.hunks, strict=True).text


@dataclass
class EditOutcome:
    attempts: int
    output_tokens: int       # every attempt the model wrote
    context_tokens: int      # failed attempts plus error messages left in the context


def land_edit(fmt: str, text: str, line: int, p: float, rng: random.Random, max_tries: int = 100) -> EditOutcome:
    """Retry until the edit lands. Each attempt slips with probability 1 - p."""
    out = ctx = 0
    for k in range(1, max_tries + 1):
        body = payload(fmt, text, line, slip=rng.random() >= p)
        out += count_tokens(body)
        try:
            result = apply(fmt, text, body)
        except EditError as e:
            ctx += count_tokens(body) + count_tokens(str(e))
            continue
        assert result == target_edit(text, line), "an applier accepted a wrong edit"
        return EditOutcome(k, out, ctx)
    raise RuntimeError("edit never landed")


def clean_cost(fmt: str, text: str, line: int) -> int:
    """m_F: output tokens of one clean attempt."""
    return count_tokens(payload(fmt, text, line, slip=False))
