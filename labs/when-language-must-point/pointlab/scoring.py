"""Losses, rewards and metrics for grounding outputs.

Two groups. The first scores one coordinate distribution: cross-entropy against an IoU reward (the post's
Figure 5). The second scores whole predictions: gIoU and cIoU for masks, boundary IoU, the unrelated
generalised IoU for boxes, and one-to-one matching for answers with many regions or none.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .geometry import Box, box_iou, interval_iou, mask_iou


# ---- one coordinate: cross-entropy vs an IoU reward --------------------------------------------

def cross_entropy(p: dict[int, float], true_bin: int) -> float:
    """-log p(true bin). Nothing else about p matters."""
    return -math.log(p.get(true_bin, 0.0))


def ce_logit_grad_norm(p: dict[int, float], true_bin: int) -> float:
    """|| d CE / d logits || = || p - onehot ||. It sees how much mass is off target, not where it went."""
    v = {k: q for k, q in p.items()}
    v[true_bin] = v.get(true_bin, 0.0) - 1.0
    return math.sqrt(sum(q * q for q in v.values()))


def edge_iou(x2: int, true: tuple[int, int]) -> float:
    """IoU of the box [x1, x2] with the true box [x1, x2*], when only the right edge is predicted."""
    return interval_iou((true[0], x2), true)


def expected_iou_reward(p: dict[int, float], true: tuple[int, int]) -> float:
    """E_{x2 ~ p}[IoU]: what a policy-gradient method maximises when the reward is IoU."""
    return sum(q * edge_iou(k, true) for k, q in p.items())


# ---- masks: two averages, and the boundary -----------------------------------------------------

def giou_ciou(preds: list[np.ndarray], gts: list[np.ndarray]) -> tuple[float, float]:
    """gIoU: mean of per-example IoUs. cIoU: total intersection over total union (large objects dominate)."""
    inters = [np.logical_and(p, g).sum() for p, g in zip(preds, gts)]
    unions = [np.logical_or(p, g).sum() for p, g in zip(preds, gts)]
    g = float(np.mean([i / u if u else 1.0 for i, u in zip(inters, unions)]))
    c = float(sum(inters) / sum(unions)) if sum(unions) else 1.0
    return g, c


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion with a (2r+1) square, written with shifts so it needs only numpy."""
    m = np.pad(mask, r, constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    h, w = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out &= m[r + dy:r + dy + h, r + dx:r + dx + w]
    return out


def boundary_iou(pred: np.ndarray, gt: np.ndarray, d: int = 2) -> float:
    """IoU computed only on the inner band of width d along each contour (Cheng et al., 2021)."""
    pb = pred & ~_erode(pred, d)
    gb = gt & ~_erode(gt, d)
    return mask_iou(pb, gb)


def generalised_iou(a: Box, b: Box) -> float:
    """GIoU from Rezatofighi et al. (2019): IoU minus the share of the enclosing box that neither box covers.
    A box loss and metric, unrelated to the segmentation 'gIoU' above."""
    iou = box_iou(a, b)
    ex = (max(a[2], b[2]) - min(a[0], b[0])) * (max(a[3], b[3]) - min(a[1], b[1]))
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return iou - (ex - union) / ex


# ---- many regions, or none ---------------------------------------------------------------------

def match_f1(pred: list, gt: list, iou_fn=box_iou, thr: float = 0.5) -> dict:
    """One-to-one matching that maximises the number of pairs with IoU >= thr, then precision, recall, F1.

    Exhaustive over assignments, which is fine for the handful of regions in one answer. An empty
    prediction for an empty target scores F1 = 1: "there is no such region" is a correct answer.
    """
    if not pred and not gt:
        return {"tp": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred or not gt:
        return {"tp": 0, "precision": 0.0 if pred else 1.0, "recall": 0.0 if gt else 1.0, "f1": 0.0}
    small, large, flip = (pred, gt, False) if len(pred) <= len(gt) else (gt, pred, True)
    best = 0
    for perm in itertools.permutations(range(len(large)), len(small)):
        tp = sum(1 for i, j in enumerate(perm)
                 if (iou_fn(large[j], small[i]) if flip else iou_fn(small[i], large[j])) >= thr)
        best = max(best, tp)
    p, r = best / len(pred), best / len(gt)
    return {"tp": best, "precision": p, "recall": r, "f1": 0.0 if best == 0 else 2 * p * r / (p + r)}


def sequence_f1(pred: list, gt: list, iou_fn=box_iou, thr: float = 0.5) -> dict:
    """The wrong way: compare the i-th predicted region with the i-th true region."""
    tp = sum(1 for a, b in zip(pred, gt) if iou_fn(a, b) >= thr)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(gt) if gt else 1.0
    return {"tp": tp, "precision": p, "recall": r, "f1": 0.0 if tp == 0 else 2 * p * r / (p + r)}
