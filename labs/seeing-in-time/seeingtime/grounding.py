"""Temporal grounding metrics, and the ceiling that frame spacing puts on them."""

from __future__ import annotations

import torch


def tiou(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    """max(0, min(ep, eg) - max(sp, sg)) / (max(ep, eg) - min(sp, sg))."""
    (sp, ep), (sg, eg) = pred, gt
    inter = max(0.0, min(ep, eg) - max(sp, sg))
    union = max(ep, eg) - min(sp, sg)
    return inter / union if union > 0 else 0.0


def recall_at_1(ious, threshold: float) -> float:
    ious = list(ious)
    return sum(i >= threshold for i in ious) / len(ious)


def frame_grid(step: float, duration: float) -> list[float]:
    """Frame times 0, step, 2*step, ... up to the duration (as in the post's Figure 9)."""
    n = int(round(duration / step))
    return [k * step for k in range(n + 1)]


def best_on_grid(gt: tuple[float, float], step: float, duration: float) -> float:
    """The best tIoU any prediction can reach if both boundaries must fall on sampled frames."""
    g = frame_grid(step, duration)
    return max(tiou((a, b), gt) for i, a in enumerate(g) for b in g[i + 1:])


def snap_nearest(gt: tuple[float, float], step: float, offset: float = 0.0) -> tuple[float, float]:
    """A perfect model that can only answer with frame times: each boundary goes to its nearest frame."""
    s = round((gt[0] - offset) / step) * step + offset
    e = round((gt[1] - offset) / step) * step + offset
    return s, e


def worst_case_floor(length: float, spacing: float) -> float:
    """Both boundaries move inward by up to half the spacing: tIoU >= (L - spacing) / L."""
    return max(0.0, (length - spacing) / length)


def snapped_oracle_scores(step: float, n: int = 20000, seed: int = 0, length=(2.0, 10.0), duration=60.0):
    """Random events, snapped by a nearest-frame oracle with a random frame phase. Returns all tIoUs."""
    g = torch.Generator().manual_seed(seed)
    L = torch.empty(n).uniform_(*length, generator=g)
    s = torch.rand(n, generator=g) * (duration - L)
    phase = torch.rand(n, generator=g) * step
    out = []
    for i in range(n):
        gt = (float(s[i]), float(s[i] + L[i]))
        out.append(tiou(snap_nearest(gt, step, float(phase[i])), gt))
    return out
