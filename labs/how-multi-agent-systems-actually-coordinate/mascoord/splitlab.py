"""The post's split-decision lab (Figure 8), ported line for line from the page's script.

    Q_MAS / Q_SAS ~ [eta(L/N) / eta(L)] x [ensemble gain capped by rho] x (1 - delta)^d

It is a toy model with made-up constants, as the post says. Porting it lets a reader sweep
it, test it, and check that the eight preset scenarios land on the verdicts the captions
describe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .voting import p_majority

P0 = 0.62  # per-agent accuracy the page assumes


def eta(x: float) -> float:
    """Context utilisation as a function of context pressure x in [0, 1]."""
    return 1.0 / (1.0 + 2.9 * x * x)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class Scenario:
    name: str
    N: int
    press: float   # context pressure of the whole task on one agent
    rho: float     # error correlation
    delta: float   # fraction of relevant information lost per handoff
    f: float       # parallelisable fraction
    tier: float    # worker price relative to the planner


def evaluate(s: Scenario) -> dict:
    n = max(1, round(s.N))
    press = _clamp(s.press, 0, 1)
    rho = _clamp(s.rho, 0, 0.95)
    delta = _clamp(s.delta, 0, 0.45)
    f = _clamp(s.f, 0.1, 0.99)
    tier = _clamp(s.tier, 0.05, 1)

    per_agent = press / n**0.75                       # 1. shorter contexts per agent
    ctx_gain = eta(per_agent) / eta(press)
    ens = rho * P0 + (1 - rho) * p_majority(n, P0)    # 2. ensemble gain, capped by rho
    ens_gain = 1.0 if n == 1 else ens / P0
    depth = 0 if n == 1 else math.ceil(math.log(n) / math.log(4)) + 1   # 3. handoffs
    handoff = (1 - delta) ** depth
    quality = ctx_gain * ens_gain * handoff

    comm = 0.16 * n                                   # one planner at price 1, workers at `tier`
    cost = 1 + (n - 1) * tier + comm * tier
    wall = (1 - f) + f / n + 0.012 * n**1.5
    return {"N": n, "ctx_gain": ctx_gain, "ens_gain": ens_gain, "handoff": handoff,
            "depth": depth, "quality": quality, "cost": cost, "wall": wall,
            "per_dollar": quality / cost}


def verdict(r: dict, s: Scenario) -> str:
    """Same rule order as the page."""
    if r["N"] == 1:
        return "Stay solo."
    if r["wall"] > 1.0:
        return "Past the optimum."
    if r["handoff"] < 0.62 and r["ctx_gain"] > 1.4:
        return "Fix the protocol."
    if r["quality"] < 1.0:
        return "Do not split."
    if r["per_dollar"] < 0.45:
        return "Not worth it."
    if s.tier <= 0.15 and r["quality"] > 1.25:
        return "Split on economics."
    if r["wall"] < 0.6:
        return "Split for latency."
    if r["ctx_gain"] > 1.25:
        return "Split for context."
    return "Split, carefully."


PRESETS = [
    Scenario("the task fits", 1, 0.35, 0.45, 0.12, 0.7, 0.2),
    Scenario("multi-hop QA, same model x4", 4, 0.3, 0.6, 0.16, 0.5, 1),
    Scenario("six-agent debate", 6, 0.28, 0.7, 0.2, 0.35, 1),
    Scenario("breadth-first research", 5, 0.92, 0.25, 0.1, 0.9, 0.25),
    Scenario("author + clean reviewer", 2, 0.88, 0.2, 0.06, 0.3, 1),
    Scenario("frontier planner, cheap workers", 8, 0.95, 0.3, 0.11, 0.85, 0.08),
    Scenario("twenty agents, serial spine", 20, 0.7, 0.5, 0.18, 0.75, 0.4),
    Scenario("good topology, bad messages", 6, 0.9, 0.25, 0.34, 0.85, 0.15),
]
