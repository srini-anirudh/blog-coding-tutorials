"""Verifiable visual rewards. Every function maps (prediction, ground truth) -> float in [0, 1].

These are plain Python and NumPy on purpose: a reward is the specification of
what we want, so it should be small enough to read in one sitting and tested
like any other function.
"""

from __future__ import annotations

import json
import re

import numpy as np

_NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                 "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, map number words to digits."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    words = [_NUMBER_WORDS.get(w, w) for w in text.split() if w not in {"a", "an", "the"}]
    return " ".join(words)


# ---------- VQA ----------

def exact_match(pred: str, gold: str) -> float:
    """R = 1[normalize(y) == normalize(y*)]."""
    return float(normalize(pred) == normalize(gold))


def contains_match(pred: str, gold: str) -> float:
    """A WEAK reward: 1 if the gold answer appears anywhere. Listing every answer games it."""
    return float(normalize(gold) in normalize(pred).split())


# ---------- boxes ----------

Box = tuple[float, float, float, float]  # (x0, y0, x1, y1)


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two axis-aligned boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def center_in_box(pred: Box, gold: Box) -> float:
    """A WEAK reward: 1 if the gold centre lies inside the predicted box. A full-image box games it."""
    cx, cy = (gold[0] + gold[2]) / 2, (gold[1] + gold[3]) / 2
    return float(pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3])


def parse_box(text: str) -> Box | None:
    """Read the first four numbers as (x0, y0, x1, y1); None if the format is wrong."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) < 4:
        return None
    x0, y0, x1, y1 = map(float, nums[:4])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def box_reward(text: str, gold: Box) -> float:
    """IoU of the parsed box; 0 for an unparseable answer (format is part of the task)."""
    box = parse_box(text)
    return 0.0 if box is None else box_iou(box, gold)


# ---------- masks ----------

def mask_iou(pred: np.ndarray, gold: np.ndarray) -> float:
    """|P ∩ G| / |P ∪ G| for boolean masks of shape (H, W). Two empty masks agree perfectly."""
    pred, gold = pred.astype(bool), gold.astype(bool)
    union = np.logical_or(pred, gold).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gold).sum() / union)


def mask_dice(pred: np.ndarray, gold: np.ndarray) -> float:
    """2|P ∩ G| / (|P| + |G|). Dice = 2 IoU / (1 + IoU), so it is always >= IoU."""
    pred, gold = pred.astype(bool), gold.astype(bool)
    total = pred.sum() + gold.sum()
    return 1.0 if total == 0 else float(2 * np.logical_and(pred, gold).sum() / total)


# ---------- OCR ----------

def levenshtein(a: str, b: str) -> int:
    """Edit distance with one rolling row: O(len(a) * len(b)) time, O(len(b)) memory."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gold: str) -> float:
    """Character error rate = edits / len(gold). Can exceed 1 for long garbage."""
    return levenshtein(pred, gold) / max(1, len(gold))


def anls(pred: str, golds: str | list[str], tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA). Similarity below 1 - tau scores 0."""
    golds = [golds] if isinstance(golds, str) else golds
    best = 0.0
    p = " ".join(pred.lower().split())
    for g in golds:
        g = " ".join(g.lower().split())
        nl = levenshtein(p, g) / max(1, len(p), len(g))
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best


# ---------- structured extraction ----------

def field_accuracy(pred_json: str, gold: dict[str, str]) -> float:
    """Fraction of gold fields whose value matches exactly after normalisation.

    Invalid JSON scores 0, and extra keys earn nothing: the model cannot pad its
    way to reward.
    """
    try:
        pred = json.loads(pred_json)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    if not isinstance(pred, dict) or not gold:
        return 0.0
    hits = sum(normalize(str(pred.get(k, ""))) == normalize(str(v)) for k, v in gold.items())
    return hits / len(gold)


# ---------- captions: the reward-hacking pair ----------

def mentioned(text: str, vocabulary: list[str]) -> set[str]:
    words = set(normalize(text).split())
    return {v for v in vocabulary if v in words}


def caption_recall(text: str, true_items: set[str], vocabulary: list[str]) -> float:
    """A WEAK reward: fraction of true items mentioned. Mentioning everything maximises it."""
    if not true_items:
        return 0.0
    return len(mentioned(text, vocabulary) & true_items) / len(true_items)


def caption_f1(text: str, true_items: set[str], vocabulary: list[str]) -> float:
    """The fixed reward: F1 of mentioned vs true items. Hallucinated mentions cost precision."""
    said = mentioned(text, vocabulary)
    hit = len(said & true_items)
    if hit == 0:
        return 0.0
    precision, recall = hit / len(said), hit / len(true_items)
    return 2 * precision * recall / (precision + recall)
