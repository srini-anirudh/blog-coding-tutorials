"""Verifiable output-format constraints, for measuring format adherence exactly.

Each constraint has an instruction sentence that is appended to the user prompt, a function that
rewrites a reference response so it satisfies the constraint, and a checker. The checker is the
ground truth for "did the model follow the format". It looks only at the shape of the answer, not
at whether the content is right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.split(text.strip().replace("\n", " ")) if s.strip()]


def _lower(r):
    return r.lower()


def _chk_lower(r):
    return r == r.lower() and any(c.isalpha() for c in r)


def _sign(r):
    return r.strip() + " Hope this helps."


def _chk_sign(r):
    return r.strip().endswith("Hope this helps.")


def _one(r):
    s = sentences(r)
    return s[0] if s else r


def _chk_one(r):
    r = r.strip()
    return bool(r) and "\n" not in r and len(sentences(r)) == 1 and r[-1] in ".!?"


def _bullets(r):
    s = sentences(r)
    return "\n".join("- " + x for x in s[:3])


def _chk_bullets(r):
    lines = [x for x in r.strip().split("\n")]
    return len(lines) == 3 and all(x.startswith("- ") and len(x) > 2 for x in lines)


def _quote(r):
    return '"' + r.strip() + '"'


def _chk_quote(r):
    r = r.strip()
    return len(r) > 2 and r[0] == '"' and r[-1] == '"'


def _sure(r):
    return "Sure, " + r.strip()


def _chk_sure(r):
    return r.strip().startswith("Sure,")


@dataclass(frozen=True)
class Constraint:
    name: str
    text: str
    apply: Callable[[str], str]
    check: Callable[[str], bool]
    min_sentences: int = 1


TRAIN = [
    Constraint("lowercase", "Answer in all lowercase.", _lower, _chk_lower),
    Constraint("sign_off", "End your answer with the exact words 'Hope this helps.'", _sign, _chk_sign),
    Constraint("one_sentence", "Answer in a single sentence.", _one, _chk_one),
    Constraint("three_bullets", "Format your answer as a list of exactly three lines, each starting with '- '.",
               _bullets, _chk_bullets, 3),
    Constraint("quoted", "Wrap your whole answer in double quotation marks.", _quote, _chk_quote),
]
HELD_OUT = Constraint("starts_sure", "Begin your answer with the word 'Sure,'.", _sure, _chk_sure)
BY_NAME = {c.name: c for c in TRAIN + [HELD_OUT]}


def make_example(ex: dict, c: Constraint) -> dict:
    """Prompt with the constraint appended; the reference response rewritten to satisfy it."""
    return {**ex, "prompt": ex["prompt"] + " " + c.text, "response": c.apply(ex["response"]), "constraint": c.name}


def eligible(ex: dict, c: Constraint) -> bool:
    return len(sentences(ex["response"])) >= c.min_sentences and 3 <= len(ex["response"]) <= 600
