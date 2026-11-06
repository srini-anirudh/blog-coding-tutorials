"""Scoring rules from the post, section "How metrics score an answer".

C8 (`c08-vlm-training/vlmtrain/metrics.py`) already builds Levenshtein distance, ANLS, relaxed
accuracy, box IoU and the click test. We import those instead of rebuilding them, and add the
rules the post explains that C8 does not have: the official VQA normaliser and the ten-subset
VQA accuracy, Acc@0.5, gIoU versus cIoU on masks, the POPE yes/no statistics and CHAIR.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

_C08 = Path(__file__).resolve().parents[3] / "c08-vlm-training"
if str(_C08) not in sys.path:
    sys.path.insert(0, str(_C08))

from vlmtrain.metrics import anls, box_iou, levenshtein, point_in_box, relaxed_accuracy  # noqa: E402

__all__ = [
    "anls", "box_iou", "levenshtein", "point_in_box", "relaxed_accuracy",
    "vqa_normalize", "vqa_accuracy", "acc_at_iou", "mask_iou", "giou_ciou", "pope_stats", "chair",
]

# --- VQA accuracy ----------------------------------------------------------------------------

# A subset of the official VQA evaluation tables. The full lists are longer (about 150
# contractions); these cover the answers in this lab and every case the tests check.
_CONTRACTIONS = {
    "dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't", "cant": "can't",
    "wont": "won't", "im": "i'm", "ive": "i've", "youre": "you're", "thats": "that's",
    "whats": "what's", "theres": "there's", "shouldnt": "shouldn't", "wasnt": "wasn't",
}
_NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@",
          "`", ",", "?", "!"]
_PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")       # a full stop that is not a decimal point
_COMMA_NUM = re.compile(r"(\d)(,)(\d)")             # "1,250" keeps its comma until the end


def vqa_normalize(answer: str) -> str:
    """The official VQA answer normaliser, in the same order as the reference code.

    1. Lower-case and trim. 2. Remove punctuation (a space replaces it unless it sits between
    letters, and a comma inside a number is dropped). 3. Drop full stops that are not decimal
    points. 4. Map number words to digits, drop articles, restore common contractions.
    """
    text = answer.replace("\n", " ").replace("\t", " ").strip().lower()
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or _COMMA_NUM.search(text) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD.sub("", out)
    words = []
    for w in out.split():
        w = _NUMBERS.get(w, w)
        if w in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def vqa_accuracy(pred: str, human_answers: list[str]) -> float:
    """VQA accuracy against ten human answers, computed the way the official code does.

    For each of the ten humans, leave that human out and score min(#matches among the other
    nine / 3, 1). Average the ten scores. An answer given by exactly three people scores 0.9,
    not 1, because in three of the ten subsets only two matching people remain.
    """
    p = vqa_normalize(pred)
    gts = [vqa_normalize(a) for a in human_answers]
    scores = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        scores.append(min(sum(g == p for g in others) / 3.0, 1.0))
    return sum(scores) / len(scores)


# --- Regions ---------------------------------------------------------------------------------

def acc_at_iou(pred_boxes, gold_boxes, threshold: float = 0.5) -> float:
    """RefCOCO-style Acc@0.5: the share of predictions whose IoU with the gold box exceeds 0.5."""
    hits = [box_iou(p, g) > threshold for p, g in zip(pred_boxes, gold_boxes, strict=True)]
    return sum(hits) / len(hits)


def mask_iou(pred: torch.Tensor, gold: torch.Tensor) -> tuple[int, int]:
    """(intersection, union) pixel counts for two boolean masks of the same shape."""
    pred, gold = pred.bool(), gold.bool()
    return int((pred & gold).sum()), int((pred | gold).sum())


def giou_ciou(preds: list[torch.Tensor], golds: list[torch.Tensor]) -> tuple[float, float]:
    """gIoU is the mean of per-image IoUs. cIoU is total intersection over total union.

    gIoU counts every image equally. cIoU counts every pixel equally, so large objects decide it.
    """
    pairs = [mask_iou(p, g) for p, g in zip(preds, golds, strict=True)]
    giou = sum(i / u if u else 1.0 for i, u in pairs) / len(pairs)
    ciou = sum(i for i, _ in pairs) / max(1, sum(u for _, u in pairs))
    return giou, ciou


# --- Hallucination ---------------------------------------------------------------------------

def yes_no(text: str) -> str | None:
    """POPE's answer rule: 'no' if the answer contains a negation word, else 'yes' if it says yes."""
    words = re.findall(r"[a-z']+", text.lower())
    if any(w in ("no", "not") for w in words):
        return "no"
    if "yes" in words:
        return "yes"
    return None


def pope_stats(preds: list[str | None], golds: list[str]) -> dict[str, float]:
    """Accuracy, precision, recall and F1 for 'yes' as the positive class, plus the yes-ratio.

    An unparsable answer counts as 'no', as in the POPE reference code. On a balanced set a
    yes-ratio far above 0.5 is the signature of a model that claims absent objects.
    """
    p = [x == "yes" for x in preds]
    g = [x == "yes" for x in golds]
    tp = sum(a and b for a, b in zip(p, g))
    fp = sum(a and not b for a, b in zip(p, g))
    fn = sum(b and not a for a, b in zip(p, g))
    n = len(g)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": sum(a == b for a, b in zip(p, g)) / n, "precision": precision,
            "recall": recall, "f1": f1, "yes_ratio": sum(p) / n, "n": n}


def chair(mentioned: list[set[str]], present: list[set[str]]) -> dict[str, float]:
    """CHAIR_i: share of mentioned objects that are absent. CHAIR_s: share of captions with any."""
    absent = [m - g for m, g in zip(mentioned, present, strict=True)]
    n_mentions = sum(len(m) for m in mentioned)
    return {"chair_i": sum(len(a) for a in absent) / max(1, n_mentions),
            "chair_s": sum(bool(a) for a in absent) / len(mentioned)}
