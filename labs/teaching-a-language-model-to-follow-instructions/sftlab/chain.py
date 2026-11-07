"""A task where serial depth matters: the running sum of n digits modulo 10.

Direct answer: the sum mod 10 in one forward pass. Reasoning trace: write the running total after every digit, so each
written token needs one addition mod 10. The checker knows every correct intermediate value."""

from __future__ import annotations

import random
import re

MOD = 10


def make_question(digits: list[int]) -> str:
    return "Add these digits and give the total modulo 10: " + " ".join(map(str, digits))


def totals(digits: list[int]) -> list[int]:
    t, out = 0, []
    for d in digits:
        t = (t + d) % MOD
        out.append(t)
    return out


def direct_answer(digits: list[int]) -> str:
    return f"Answer: {totals(digits)[-1]}"


def trace_answer(digits: list[int]) -> str:
    t = totals(digits)
    return "Running total: " + ", ".join(map(str, t)) + f". Answer: {t[-1]}"


def make_example(n: int, rng: random.Random, kind: str) -> dict:
    d = [rng.randrange(10) for _ in range(n)]
    return {"prompt": make_question(d), "response": direct_answer(d) if kind == "direct" else trace_answer(d),
            "digits": d, "n": n}


TRACE = re.compile(r"Running total:\s*([\d,\s]*)\.\s*Answer:\s*(\d+)")
ANS = re.compile(r"Answer:\s*(\d+)")


def parse_final(reply: str):
    m = ANS.search(reply)
    return int(m.group(1)) if m else None


def check_trace(reply: str, digits: list[int]) -> dict:
    """Final answer correct? Every written step correct given the previous WRITTEN value? Format valid?"""
    gold = totals(digits)
    final = parse_final(reply)
    m = TRACE.search(reply)
    if not m:
        return {"final_ok": final == gold[-1], "valid": False, "steps_ok": 0, "steps": len(digits), "format": False}
    vals = [int(x) for x in m.group(1).replace(" ", "").split(",") if x != ""]
    steps_ok, prev = 0, 0
    for i, v in enumerate(vals[:len(digits)]):
        steps_ok += v == (prev + digits[i]) % MOD
        prev = v
    valid = len(vals) == len(digits) and steps_ok == len(digits) and final == vals[-1]
    return {"final_ok": final == gold[-1], "valid": valid, "steps_ok": steps_ok, "steps": len(digits), "format": True}
