"""Writing a box as tokens: eight published conventions, their parsers, and the cost of a coordinate grid.

Each convention is a pair of functions. `encode` turns a normalised box (x1, y1, x2, y2 in [0, 1]) into the
string the model would write. `parse` turns that string back into a normalised box. The rounding rule of each
system follows its paper or official code where they state it; where they do not, we round to the nearest bin.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

NBox = tuple[float, float, float, float]


def _q(v: float, n: int) -> int:
    """Nearest integer on a 0..n grid. round() on v * n, with a tiny nudge so 0.5 cases do not depend on float noise."""
    return int(round(v * n + 1e-9))


@dataclass(frozen=True)
class Convention:
    name: str
    encode: Callable[[NBox], str]
    parse: Callable[[str], NBox]
    special_tokens: tuple[str, ...] = ()   # tokens added to the vocabulary (one token each)
    note: str = ""


def _nums(s: str) -> list[float]:
    return [float(m) for m in re.findall(r"\d+(?:\.\d+)?", s)]


# Qwen2.5-VL writes pixels of the resized image it was given. The post's example uses an 868 x 644 input.
QWEN25_SIZE = (868, 644)


def _qwen25_encode(b: NBox, size=QWEN25_SIZE) -> str:
    w, h = size
    box = [_q(b[0], w), _q(b[1], h), _q(b[2], w), _q(b[3], h)]
    return json.dumps({"bbox_2d": box, "label": "dog"})


def _qwen25_parse(s: str, size=QWEN25_SIZE) -> NBox:
    w, h = size
    x1, y1, x2, y2 = json.loads(s)["bbox_2d"]
    return (x1 / w, y1 / h, x2 / w, y2 / h)


def _molmo_encode(b: NBox) -> str:
    x, y = (b[0] + b[2]) / 2 * 100, (b[1] + b[3]) / 2 * 100
    return f'<point x="{x + 1e-9:.1f}" y="{y + 1e-9:.1f}" alt="dog">dog</point>'


def _molmo_parse(s: str) -> NBox:
    x, y = (float(v) / 100 for v in re.findall(r'[xy]="([\d.]+)"', s))
    return (x, y, x, y)  # a point: a box with zero extent


def _paligemma_encode(b: NBox) -> str:
    # 1,024 bins, y before x. floor(v * 1024) matches the strings in the post's table.
    k = [min(1023, int(v * 1024)) for v in (b[1], b[0], b[3], b[2])]
    return "".join(f"<loc{v:04d}>" for v in k) + " dog"


def _paligemma_parse(s: str) -> NBox:
    y1, x1, y2, x2 = (int(v) / 1024 for v in re.findall(r"<loc(\d{4})>", s)[:4])
    return (x1, y1, x2, y2)


CONVENTIONS: list[Convention] = [
    Convention("Shikra",
               lambda b: "[" + ",".join(f"{v:.3f}" for v in b) + "]",
               lambda s: tuple(_nums(s)[:4]),
               note="plain-text decimals in [0, 1], three places"),
    Convention("Qwen-VL",
               lambda b: f"<ref>the dog</ref><box>({_q(b[0], 1000)},{_q(b[1], 1000)}),({_q(b[2], 1000)},{_q(b[3], 1000)})</box>",
               lambda s: tuple(v / 1000 for v in _nums(s.split("<box>")[1])[:4]),
               special_tokens=("<ref>", "</ref>", "<box>", "</box>"),
               note="plain-text integers on a 0-1000 grid, special tags"),
    Convention("CogVLM",
               lambda b: "[[" + ",".join(f"{_q(v, 1000):03d}" for v in b) + "]]",
               lambda s: tuple(v / 1000 for v in _nums(s)[:4]),
               note="three-digit integers 000-999"),
    Convention("MiniGPT-v2",
               lambda b: "{" + "".join(f"<{_q(v, 100)}>" for v in b) + "}",
               lambda s: tuple(v / 100 for v in _nums(s)[:4]),
               note="integers on a 0-100 grid (plain text)"),
    Convention("Florence-2",
               lambda b: "".join(f"<loc_{_q(v, 1000)}>" for v in b),
               lambda s: tuple(int(v) / 1000 for v in re.findall(r"<loc_(\d+)>", s)[:4]),
               special_tokens=tuple(f"<loc_{k}>" for k in range(1000)),
               note="1,000 dedicated location tokens"),
    Convention("PaliGemma", _paligemma_encode, _paligemma_parse,
               special_tokens=tuple(f"<loc{k:04d}>" for k in range(1024)),
               note="1,024 location tokens, y before x"),
    Convention("Qwen2.5-VL", _qwen25_encode, _qwen25_parse,
               note="absolute pixels of the resized input (868 x 644)"),
    Convention("Molmo (point)", _molmo_encode, _molmo_parse,
               note="a point, plain-text decimals on a 0-100 scale"),
]

BY_NAME = {c.name: c for c in CONVENTIONS}


def digit_split_count(s: str, special_tokens=()) -> int:
    """Token count under a simple digit-splitting tokenizer: special tokens and words count once,
    every digit and every punctuation mark counts once. Real BPE tokenizers differ in detail."""
    specials = sorted(special_tokens, key=len, reverse=True)
    pattern = ("|".join(re.escape(t) for t in specials) + "|" if specials else "") + r"\d|[A-Za-z_]+|\S"
    return len(re.findall(pattern, s))


# ---- the cost of a coordinate grid --------------------------------------------------------------

def bin_width(image_px: float, bins: int) -> float:
    """Delta = W / N, in pixels of the original image."""
    return image_px / bins


def worst_case_iou(obj_px: float, image_px: float, bins: int) -> float:
    """IoU_min = ((s - Delta) / s)^2 for a square of side s whose four edges each move inward by Delta / 2."""
    d = bin_width(image_px, bins)
    return max(0.0, (obj_px - d) / obj_px) ** 2


def brute_force_worst_iou(obj_px: float, delta: float, steps: int = 9) -> tuple[float, tuple]:
    """Try every combination of edge errors in [-Delta/2, Delta/2] on a grid, return the smallest IoU and its errors."""
    from .geometry import box_iou

    errs = np.linspace(-delta / 2, delta / 2, steps)
    true = (0.0, 0.0, obj_px, obj_px)
    best = (2.0, None)
    for e in itertools.product(errs, repeat=4):
        x1, y1, x2, y2 = e[0], e[1], obj_px + e[2], obj_px + e[3]
        iou = box_iou(true, (x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else 0.0
        if iou < best[0]:
            best = (float(iou), tuple(round(float(v), 3) for v in e))
    return best


def quantise(v_px: float, image_px: float, bins: int) -> float:
    """Round a pixel coordinate to the centre of its bin (what a perfect model on this grid can say)."""
    d = bin_width(image_px, bins)
    k = min(bins - 1, max(0, int(v_px // d)))
    return (k + 0.5) * d


def digit_error_shift(true_value: int, wrong_value: int, grid: int = 1000) -> float:
    """How far an edge moves, as a share of the image, when one digit of a 0-1000 coordinate is wrong."""
    return abs(wrong_value - true_value) / grid


# ---- Qwen's resize: what "absolute pixels" means -------------------------------------------------

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56,
                 max_pixels: int = 14 * 14 * 4 * 16384) -> tuple[int, int]:
    """The Qwen2-VL / Qwen2.5-VL resize rule, rewritten from the public processor code: both sides become
    multiples of `factor` (patch 14 x merge 2 = 28) and the pixel count is kept inside [min_pixels, max_pixels].
    Qwen3-VL uses patch 16, so factor 32. Returns (height, width) of the image the model actually sees."""
    import math
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w
