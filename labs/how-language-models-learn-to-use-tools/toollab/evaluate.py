"""Score a decoded call factor by factor: call at all, which tool, which arguments.

The post writes p(a* | c) = p(call?) * p(which tool? | call) * p(which args? | tool). Each factor gets its
own number here, so one "function-calling accuracy" does not hide three different skills.
"""

from __future__ import annotations

import random
from collections import Counter

from .world import Example, Tool, parse_call


def score(ex: Example, out: list[str], train_names: set[str] | None = None) -> dict:
    call = parse_call(out)
    gold_none = ex.gold is None
    r = {"parse": call is not None, "exact": out == ex.target_tokens(), "gold_none": gold_none}
    pred_none = bool(call and call.get("none"))
    r["decide"] = call is not None and pred_none == gold_none
    if not gold_none and call is not None and not pred_none:
        r["select"] = call["name"] == ex.gold.name_str
        r["in_context"] = call["name"] in {t.name_str for t in ex.tools}
        r["train_name"] = bool(train_names) and call["name"] in train_names
        if r["select"]:
            r["args"] = call["args"] == ex.args
    return r


def summarise(rows: list[dict]) -> dict:
    """Conditional rates, matching the three factors of the chain rule."""
    def rate(key: str, cond=lambda r: True) -> float | None:
        sel = [r[key] for r in rows if cond(r) and key in r]
        return round(sum(sel) / len(sel), 4) if sel else None

    calls = [r for r in rows if not r["gold_none"]]
    wrong_name = [r for r in calls if r.get("select") is False]
    return {
        "n": len(rows),
        "exact": rate("exact"),
        "parse": rate("parse"),
        "decide": rate("decide"),  # factor 1: call vs NONE, on every example
        "abstain_when_none": rate("decide", lambda r: r["gold_none"]),
        "select_given_call": rate("select"),  # factor 2, among gold calls where the model called
        "args_given_tool": rate("args"),  # factor 3, among calls with the right tool
        "exact_on_calls": round(sum(r["exact"] for r in calls) / max(1, len(calls)), 4),
        "wrong_name_not_in_context": round(sum(not r["in_context"] for r in wrong_name) / max(1, len(wrong_name)), 4)
        if wrong_name else None,
        "wrong_name_is_training_tool": round(sum(r["train_name"] for r in wrong_name) / max(1, len(wrong_name)), 4)
        if wrong_name else None,
    }


def eval_set(pool: list[Tool], n: int, seed: int, **kw) -> list[Example]:
    from .world import make_example

    rng = random.Random(seed)
    return [make_example(pool, rng, **kw) for _ in range(n)]


def count_tokens(rows: list[list[str]]) -> Counter:
    return Counter(t for r in rows for t in r)
