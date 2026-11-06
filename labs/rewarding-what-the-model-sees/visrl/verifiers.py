"""The visual verifiers of the post's taxonomy, plus the weak variants that get hacked.

Each reward maps (model output, annotation) to a float, usually in [0, 1].
Exact match, box IoU, mask IoU and Dice, CER and ANLS come from C9. This file
adds what the post adds: numeric tolerance, GIoU, coverage, thresholds, point
rewards, matched multi-object rewards, the "list every category" hack and its
length-aware fix, strict and lenient answer parsers, and a schema-gated F1.

Boxes are (x0, y0, x1, y1) in image coordinates.
"""

from __future__ import annotations

import itertools
import json
import re

from ._c09 import rewards as _c9

normalize = _c9.normalize
exact_match = _c9.exact_match
box_area = _c9.box_area
box_iou = _c9.box_iou
mask_iou = _c9.mask_iou
mask_dice = _c9.mask_dice
cer = _c9.cer
anls = _c9.anls

Box = tuple[float, float, float, float]
_NUMBER = r"-?\d+(?:\.\d+)?"


# ---------- numbers ----------

def parse_number(text: str, strict: bool = False) -> float | None:
    """The first number in the text. strict=True needs exactly one number, so "56 to 58" fails."""
    nums = re.findall(_NUMBER, text.replace(",", ""))
    if not nums or (strict and len(nums) != 1):
        return None
    return float(nums[0])


def relaxed_accuracy(pred: str, gold: str, tol: float = 0.05, strict: bool = True) -> float:
    """ChartQA's relaxed accuracy: R = 1[|y - y*| <= tol * |y*|] for numbers, exact match otherwise."""
    g = parse_number(gold, strict=True)
    if g is None:
        return exact_match(pred, gold)
    p = parse_number(pred, strict=strict)
    return float(p is not None and abs(p - g) <= tol * abs(g))


# ---------- boxes ----------

def _intersection(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def giou(pred: Box, gold: Box) -> float:
    """GIoU = IoU - |C minus (B union B*)| / |C|, with C the smallest box enclosing both. In [-1, 1]."""
    inter = _intersection(pred, gold)
    union = box_area(pred) + box_area(gold) - inter
    c = (min(pred[0], gold[0]), min(pred[1], gold[1]), max(pred[2], gold[2]), max(pred[3], gold[3]))
    area_c = box_area(c)
    if union <= 0 or area_c <= 0:
        return 0.0
    return inter / union - (area_c - union) / area_c


def coverage(pred: Box, gold: Box) -> float:
    """Fraction of the target covered. A recall-only signal: the whole image scores 1."""
    return _intersection(pred, gold) / max(box_area(gold), 1e-12)


def iou_at(pred: Box, gold: Box, threshold: float = 0.5) -> float:
    """Thresholded IoU, 1[IoU >= t]. Removes the free-credit floor; gives no signal on a miss."""
    return float(box_iou(pred, gold) >= threshold)


def size_match(pred: Box, gold: Box) -> float:
    """min(area) / max(area): 1 when the sizes agree, whatever the position. A GUI-G1-style size term."""
    a, b = box_area(pred), box_area(gold)
    return min(a, b) / max(a, b) if max(a, b) > 0 else 0.0


def center_hit(pred: Box, gold: Box) -> float:
    """GUI-G1's hit reward: does the predicted box's centre land inside the target?"""
    return point_in_box(((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2), gold)


# ---------- points ----------

def point_in_box(p: tuple[float, float], box: Box) -> float:
    return float(box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3])


def point_distance(p: tuple[float, float], target: tuple[float, float], d: float) -> float:
    """R = max(0, 1 - ||p - p*|| / d): graded credit that decays with distance."""
    dist = ((p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2) ** 0.5
    return max(0.0, 1.0 - dist / d)


# ---------- several objects ----------

def match(preds: list[Box], golds: list[Box]) -> list[tuple[int, int, float]]:
    """Hungarian matching by maximum total IoU: (pred index, gold index, IoU) per pair.

    Solved by trying every assignment, which is exact and readable for the handful
    of objects in a test. A real detector reward would use an O(n^3) solver.
    """
    if not preds or not golds:
        return []
    if max(len(preds), len(golds)) > 8:
        raise ValueError("brute-force matching is for at most 8 objects")
    iou = [[box_iou(p, g) for g in golds] for p in preds]
    best, best_pairs = -1.0, []
    if len(preds) >= len(golds):
        for perm in itertools.permutations(range(len(preds)), len(golds)):
            score = sum(iou[p][g] for g, p in enumerate(perm))
            if score > best:
                best, best_pairs = score, [(p, g) for g, p in enumerate(perm)]
    else:
        for perm in itertools.permutations(range(len(golds)), len(preds)):
            score = sum(iou[p][g] for p, g in enumerate(perm))
            if score > best:
                best, best_pairs = score, list(enumerate(perm))
    return [(p, g, iou[p][g]) for p, g in best_pairs]


def recall_precision(preds: list[Box], golds: list[Box], threshold: float = 0.5) -> tuple[float, float]:
    """The detection rewards of Vision-R1 (Zhan): recall = matched golds above t / golds,
    precision = mean IoU over predicted boxes, unmatched ones counting 0."""
    pairs = match(preds, golds)
    recall = sum(iou >= threshold for _, _, iou in pairs) / max(len(golds), 1)
    precision = sum(iou for _, _, iou in pairs) / max(len(preds), 1)
    return recall, precision


def category_score(pred: dict[str, list[Box]], gold: dict[str, list[Box]], threshold: float = 0.5) -> float:
    """A COCO-style per-category score, averaged over categories PRESENT in the image.

    For each gold category: the share of its objects matched above the threshold.
    Predicted categories with no gold objects are skipped, as COCO evaluation skips
    them. That skip is the hole VLM-R1 found: listing every category costs nothing.
    """
    if not gold:
        return 0.0
    scores = []
    for cat, boxes in gold.items():
        pairs = match(pred.get(cat, []), boxes)
        scores.append(sum(iou >= threshold for _, _, iou in pairs) / len(boxes))
    return sum(scores) / len(scores)


def length_aware(score: float, n_pred: int, n_gold: int) -> float:
    """VLM-R1's fix: multiply by min(1, L_gt / L_pred), so a list longer than the truth pays."""
    return score * min(1.0, n_gold / n_pred) if n_pred > 0 else 0.0


# ---------- multiple choice ----------

def lenient_choice(text: str, gold: str) -> float:
    """A WEAK parser: credit if the gold letter appears anywhere as a word. "B or C" passes for both."""
    return float(gold.upper() in re.findall(r"\b([A-E])\b", text.upper()))


def strict_choice(text: str, gold: str) -> float:
    """Exactly one letter, in one place: the response must end with <answer>X</answer>."""
    m = re.fullmatch(r"(?s).*<answer>\s*([A-E])\s*</answer>\s*", text)
    return float(m is not None and len(re.findall(r"<answer>", text)) == 1 and m.group(1) == gold.upper())


# ---------- structured extraction ----------

def _fields(text: str, schema: set[str]) -> set[tuple[str, str]] | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or not set(obj) <= schema or not all(isinstance(v, str) for v in obj.values()):
        return None
    return {(k, normalize(v)) for k, v in obj.items() if v.strip()}


def json_field_recall(text: str, gold: dict[str, str], schema: set[str]) -> float:
    """A WEAK reward: share of gold fields found. Filling every schema field with a guess cannot lower it."""
    said = _fields(text, schema) or set()
    truth = {(k, normalize(v)) for k, v in gold.items()}
    return len(said & truth) / max(len(truth), 1)


def json_field_f1(text: str, gold: dict[str, str], schema: set[str]) -> float:
    """R = 1[valid schema] * F1(fields(y), fields(y*)). Invalid JSON or unknown keys score 0."""
    said = _fields(text, schema)
    if said is None:
        return 0.0
    truth = {(k, normalize(v)) for k, v in gold.items()}
    hit = len(said & truth)
    if hit == 0:
        return 0.0
    precision, recall = hit / len(said), hit / len(truth)
    return 2 * precision * recall / (precision + recall)
