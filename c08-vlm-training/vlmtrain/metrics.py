"""Evaluation metrics for multimodal tasks, each small enough to check by hand.

Every function takes plain strings (or boxes) and returns a float in [0, 1], except the
error rates (CER, WER), which can exceed 1 when the prediction is much longer than the reference.
"""

from __future__ import annotations

import re
import string

_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_PUNCT = str.maketrans("", "", string.punctuation.replace(".", ""))


def normalize_answer(text: str) -> str:
    """VQA-style normalisation: lowercase, drop punctuation and articles, map number words to digits.

    The full VQAv2 normaliser has more rules (contractions, a few spelling maps). These four cover
    every answer our synthetic tasks produce, and each one is tested.
    """
    text = text.lower().strip().translate(_PUNCT)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", "", text)  # keep decimal points, drop full stops
    words = [_NUMBER_WORDS.get(w, w) for w in text.split() if w not in _ARTICLES]
    return " ".join(words)


def exact_match(pred: str, golds: str | list[str]) -> float:
    """1.0 if the normalised prediction equals any normalised gold answer."""
    golds = [golds] if isinstance(golds, str) else golds
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def levenshtein(a: list | str, b: list | str) -> int:
    """Edit distance (insertions, deletions, substitutions all cost 1). O(len(a) * len(b))."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    """Character error rate: edits needed to turn pred into ref, divided by len(ref)."""
    if not ref:
        return float(len(pred) > 0)
    return levenshtein(pred, ref) / len(ref)


def wer(pred: str, ref: str) -> float:
    """Word error rate: the same edit distance computed over whitespace-separated words."""
    r = ref.split()
    if not r:
        return float(len(pred.split()) > 0)
    return levenshtein(pred.split(), r) / len(r)


def anls(pred: str, golds: str | list[str], tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity for one question (DocVQA definition).

    score = max over golds of (1 - NL) if NL < tau else 0, with NL = lev(p, g) / max(|p|, |g|)
    on lowercased, whitespace-trimmed strings. The threshold keeps a wrong answer that happens to
    share characters with the gold one from earning partial credit.
    """
    golds = [golds] if isinstance(golds, str) else golds
    p = " ".join(pred.lower().split())
    best = 0.0
    for g in golds:
        g = " ".join(g.lower().split())
        if not p and not g:
            return 1.0
        nl = levenshtein(p, g) / max(len(p), len(g))
        best = max(best, 1.0 - nl if nl < tau else 0.0)
    return best


def parse_number(text: str) -> float | None:
    """First number in the text ("about 7.5 units" -> 7.5), or None. Commas and % are ignored."""
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", "").replace("%", ""))
    return float(m.group()) if m else None


def relaxed_accuracy(pred: str, gold: str, tol: float = 0.05) -> float:
    """ChartQA relaxed accuracy: numeric answers may be off by `tol` relative error.

    Non-numeric gold answers (a category name, "yes") fall back to exact match.
    """
    g = parse_number(gold)
    if g is None:
        return exact_match(pred, gold)
    p = parse_number(pred)
    if p is None:
        return 0.0
    if g == 0:
        return float(p == 0)
    return float(abs(p - g) / abs(g) <= tol)


Box = tuple[float, float, float, float]  # (x0, y0, x1, y1), x0 <= x1 and y0 <= y1


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two axis-aligned boxes in the same coordinate system."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def point_in_box(point: tuple[float, float], box: Box) -> float:
    """1.0 if the predicted point lands inside the gold box (the Molmo-style pointing metric)."""
    x, y = point
    return float(box[0] <= x <= box[2] and box[1] <= y <= box[3])
