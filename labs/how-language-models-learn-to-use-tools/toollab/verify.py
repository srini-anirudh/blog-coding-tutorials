"""Generate, execute, verify, keep: APIGen's three gates, and the same checks reused as an RL reward.

Gates run in order of cost, and each sees only what the cheaper gates let through:
  1. format    - the span parses, names a tool in context, and uses exactly the schema's field names
  2. execution - the backend runs it (here: every value has the type its field wants)
  3. semantic  - the call does what the request asked (APIGen uses a model as judge; this lab has the
                 ground truth, so a rule compares the call with the request's intent)

The reward is r = r_fmt + r_exec + r_match, each a 0/1 a program can compute.
"""

from __future__ import annotations

from .world import Example, execute, parse_call


def format_ok(ex: Example, call: dict | None) -> bool:
    if call is None:
        return False
    if call.get("none"):
        return True
    by_name = {t.name_str: t for t in ex.tools}
    return call["name"] in by_name and set(call["args"]) == set(by_name[call["name"]].field_types())


def execution_ok(ex: Example, call: dict) -> bool:
    if call.get("none"):
        return True  # nothing to run
    try:
        execute(call, ex.tools)
        return True
    except (KeyError, ValueError, TypeError):
        return False


def semantic_ok(ex: Example, call: dict) -> bool:
    if ex.gold is None:
        return bool(call.get("none"))
    return not call.get("none") and call["name"] == ex.gold.name_str and call["args"] == ex.args


def gates(ex: Example, out: list[str]) -> str:
    """The first gate a sample fails, or 'kept'."""
    call = parse_call(out)
    if not format_ok(ex, call):
        return "format"
    if not execution_ok(ex, call):
        return "execution"
    if not semantic_ok(ex, call):
        return "semantic"
    return "kept"


def reward(ex: Example, out: list[str]) -> dict:
    """Three 0/1 terms a program can check: it parses against the schema, it runs, it is the right call.

    The post's third term is r_outcome, "the goal state was reached". In this world each request has
    exactly one correct call and calls have no side effects, so reaching the goal and matching the
    gold call are the same event. The lab therefore scores "it ran" as the middle term instead.
    """
    call = parse_call(out)
    r_fmt = float(format_ok(ex, call))
    r_exec = float(r_fmt > 0 and execution_ok(ex, call))
    r_match = float(r_exec > 0 and semantic_ok(ex, call))
    return {"r_fmt": r_fmt, "r_exec": r_exec, "r_match": r_match, "r": r_fmt + r_exec + r_match}
