"""Mixture arithmetic: per-source caps, three ways to state a share, loss-signal share, the Pareto front.

C8 (`c08-vlm-training/vlmtrain/mixture.py`) builds the samplers: proportional, uniform, temperature.
This file adds what the post adds on top: a share in examples is not a share in compute or in loss tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    examples: int
    visual_tokens: float  # per example
    text_tokens: float    # per example, prompt and answer
    loss_tokens: float    # per example, answer tokens only


def cap(sources: list[Source], limit: int) -> list[Source]:
    """Cambrian-1 style: subsample any source above `limit` examples down to `limit`."""
    return [Source(s.name, min(s.examples, limit), s.visual_tokens, s.text_tokens, s.loss_tokens) for s in sources]


def shares(sources: list[Source]) -> dict[str, dict[str, float]]:
    """Each source's share of examples, of compute (all positions) and of loss tokens."""
    ex = {s.name: s.examples for s in sources}
    comp = {s.name: s.examples * (s.visual_tokens + s.text_tokens) for s in sources}
    loss = {s.name: s.examples * s.loss_tokens for s in sources}
    norm = lambda d: {k: v / sum(d.values()) for k, v in d.items()}  # noqa: E731
    e, c, l_ = norm(ex), norm(comp), norm(loss)
    return {s.name: {"examples": e[s.name], "compute": c[s.name], "loss_tokens": l_[s.name]} for s in sources}


def signal_share(weights: dict[str, float], loss_tokens: dict[str, float], norm: str) -> dict[str, float]:
    """Share of the gradient signal each source receives.
    token norm: proportional to weight x answer length. example norm: equal to the sampling weight."""
    raw = {k: w * (loss_tokens[k] if norm == "token" else 1.0) for k, w in weights.items()}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


def dominates(a: tuple, b: tuple) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_front(points: list[tuple]) -> list[int]:
    """Indices of the points that no other point beats on every axis (higher is better)."""
    return [i for i, p in enumerate(points) if not any(dominates(q, p) for j, q in enumerate(points) if j != i)]


def pick_weighted(points: list[tuple], w: tuple) -> int:
    return max(range(len(points)), key=lambda i: sum(a * b for a, b in zip(points[i], w)))


def pick_constrained(points: list[tuple], maximise: int, floors: dict[int, float]) -> int | None:
    """Best point on axis `maximise` among points with points[i][k] >= floors[k] for every k."""
    ok = [i for i, p in enumerate(points) if all(p[k] >= f for k, f in floors.items())]
    return max(ok, key=lambda i: points[i][maximise]) if ok else None
