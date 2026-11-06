"""A procedural arithmetic benchmark with an exact answer key and two cheap, imperfect verifiers.

Eight difficulty levels give a spread of per-problem success rates, which is what the
coverage scaling law needs. Problems are generated from a seed, so nothing is downloaded.
"""

from __future__ import annotations

import random
import re

# (name, template). Each slot is a number of digits.
LEVELS = [
    ("add 2d+2d", "{a2} + {b2}"),
    ("add 3d+3d+3d", "{a3} + {b3} + {c3}"),
    ("mul 1d*2d", "{a1} * {b2}"),
    ("mul 2d*2d", "{a2} * {b2}"),
    ("mul+add 2d*2d+3d", "{a2} * {b2} + {c3}"),
    ("mul 3d*2d", "{a3} * {b2}"),
    ("mul 3d*3d", "{a3} * {b3}"),
    ("mul 2d*2d*2d", "{a2} * {b2} * {c2}"),
]

PROMPT = ("What is {expr}? Think step by step. Then write the final answer on its own line "
          "as 'Answer: <number>'.")


def _num(rng: random.Random, digits: int) -> int:
    return rng.randint(10 ** (digits - 1) if digits > 1 else 2, 10 ** digits - 1)


def make_problems(per_level: int = 12, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for level, (name, tmpl) in enumerate(LEVELS):
        for _ in range(per_level):
            slots = {f"{v}{d}": _num(rng, d) for v in "abc" for d in (1, 2, 3)}
            expr = tmpl.format(**slots)
            out.append({"id": len(out), "level": level, "level_name": name, "expr": expr,
                        "truth": int(eval(expr)), "prompt": PROMPT.format(expr=expr)})
    return out


_INT = re.compile(r"-?\d[\d,]*")


def parse_answer(text: str) -> int | None:
    """Integer after the last 'Answer:', else in the last \\boxed{}, else the last integer, else None."""
    for key in ("Answer:", "\\boxed{"):
        i = text.rfind(key)
        if i >= 0:
            m = _INT.search(text[i + len(key):])
            if m:
                return _to_int(m.group())
    found = _INT.findall(text)
    return _to_int(found[-1]) if found else None


def _to_int(s: str) -> int | None:
    s = s.replace(",", "")
    return int(s) if s not in ("", "-") else None


def last_digit_check(ans: int | None, truth: int) -> bool:
    """A weak 'unit test': only the last digit is checked."""
    return ans is not None and abs(ans) % 10 == abs(truth) % 10


def shape_check(ans: int | None, truth: int) -> bool:
    """A stronger but still partial test: last digit and number of digits both match."""
    return last_digit_check(ans, truth) and len(str(abs(ans))) == len(str(abs(truth)))
