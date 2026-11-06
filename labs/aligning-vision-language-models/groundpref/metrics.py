"""Hallucination and abstention metrics, computed against exact scene facts."""

from __future__ import annotations

import torch

from .world import Claim, Scene


def score_response(scene: Scene, claims: list[Claim]) -> dict[str, int]:
    """Count each kind of error in one response."""
    obj_h = sum(c.obj not in scene.facts for c in claims)
    present = [c for c in claims if c.obj in scene.facts]
    return {
        "mentions": len(claims),
        "object": obj_h,
        "colour": sum(c.colour != scene.facts[c.obj].colour for c in present),
        "count": sum(c.count != scene.facts[c.obj].count for c in present),
        "present_mentions": len(present),
        "covered": len({c.obj for c in present}),
        "in_scene": len(scene.facts),
    }


def summarise(rows: list[dict[str, int]]) -> dict[str, float]:
    """CHAIR_i, CHAIR_s, object recall (coverage), and per-claim attribute and count error rates."""
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    n = len(rows)
    return {
        "chair_i": s["object"] / max(1, s["mentions"]),
        "chair_s": sum(r["object"] > 0 for r in rows) / n,
        "recall": s["covered"] / max(1, s["in_scene"]),
        "colour_err": s["colour"] / max(1, s["present_mentions"]),
        "count_err": s["count"] / max(1, s["present_mentions"]),
        "claims_per_response": s["mentions"] / n,
        "object_per_response": s["object"] / n,
        "colour_per_response": s["colour"] / n,
        "count_per_response": s["count"] / n,
    }


def risk_coverage(conf: torch.Tensor, correct: torch.Tensor, tau: float, c: float) -> dict[str, float]:
    """Answer when conf > tau. Effective reliability: +1 right, -c wrong, 0 abstain (Whitehead et al.)."""
    ans = conf > tau
    cov = ans.float().mean().item()
    risk = (1 - correct[ans].float().mean().item()) if ans.any() else 0.0
    er = (ans.float() * (correct.float() - c * (1 - correct.float()))).mean().item()
    return {"tau": tau, "coverage": cov, "risk": risk, "eff_reliability": er}


def expected_calibration_error(conf: torch.Tensor, correct: torch.Tensor, bins: int = 10) -> float:
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].float().mean().item())
    return ece
