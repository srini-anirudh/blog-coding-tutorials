"""Grounding data: points, boxes and masks, and the text formats a language model can emit.

A VLM can only "point" by writing tokens. This module turns pixel geometry into those tokens and
parses them back. Two text formats are supported for the same quantised coordinates:

    tokens  <loc_3><loc_10><loc_8><loc_15>     one new vocabulary item per coordinate (PaliGemma style)
    text    [3, 10, 8, 15]                     ordinary digits the LM already knows (Qwen-VL style)

Coordinates are quantised to `bins` integer cells over the image width and height, so both
formats carry exactly the same information. Masks are stored as run-length encodings (COCO style).
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageDraw

from .metrics import Box

LOC_BINS = 32


def loc_token(k: int) -> str:
    return f"<loc_{k}>"


def loc_tokens(bins: int = LOC_BINS) -> list[str]:
    """The new vocabulary items the tokenizer must learn for the `tokens` format."""
    return [loc_token(k) for k in range(bins)]


def quantize(v: float, size: float, bins: int = LOC_BINS) -> int:
    """Pixel coordinate in [0, size] -> integer bin in [0, bins - 1]."""
    return int(min(bins - 1, max(0, np.floor(v / size * bins))))


def dequantize(k: int, size: float, bins: int = LOC_BINS) -> float:
    """Bin index -> pixel coordinate at the bin centre."""
    return (k + 0.5) * size / bins


def encode_box(box: Box, width: float, height: float, fmt: str = "tokens", bins: int = LOC_BINS) -> str:
    """Pixel box (x0, y0, x1, y1), right/bottom edges exclusive -> grounding string.

    The far edges are quantised through the last covered pixel (x1 - 0.5), so a box that fills
    bins 0..3 encodes as 0..3 and not 0..4.
    """
    q = [quantize(box[0], width, bins), quantize(box[1], height, bins),
         quantize(box[2] - 0.5, width, bins), quantize(box[3] - 0.5, height, bins)]
    return _format(q, fmt)


def encode_point(point: tuple[float, float], width: float, height: float,
                 fmt: str = "tokens", bins: int = LOC_BINS) -> str:
    q = [quantize(point[0], width, bins), quantize(point[1], height, bins)]
    return _format(q, fmt)


def _format(q: list[int], fmt: str) -> str:
    if fmt == "tokens":
        return "".join(loc_token(k) for k in q)
    if fmt == "text":
        return "[" + ", ".join(str(k) for k in q) + "]"
    raise ValueError(f"unknown grounding format {fmt!r}")


_LOC_RE = re.compile(r"<loc_(\d+)>")
_INT_RE = re.compile(r"-?\d+")


def parse_coords(text: str, fmt: str = "tokens") -> list[int]:
    """Pull the integer bins back out of generated text. Malformed text gives fewer numbers."""
    pattern = _LOC_RE if fmt == "tokens" else _INT_RE
    return [int(m) for m in pattern.findall(text)]


def decode_box(text: str, width: float, height: float, fmt: str = "tokens",
               bins: int = LOC_BINS) -> Box | None:
    """Grounding string -> pixel box, or None when the model did not write four coordinates.

    Bin k covers [k, k+1) * size / bins, so a box from bin a to bin b spans a's left edge to b's right
    edge. Using edges (not centres) makes encode -> decode lose at most one bin per side.
    """
    q = parse_coords(text, fmt)
    if len(q) < 4:
        return None
    x0, y0, x1, y1 = (min(max(v, 0), bins - 1) for v in q[:4])
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    sx, sy = width / bins, height / bins
    return (x0 * sx, y0 * sy, (x1 + 1) * sx, (y1 + 1) * sy)


def decode_point(text: str, width: float, height: float, fmt: str = "tokens",
                 bins: int = LOC_BINS) -> tuple[float, float] | None:
    q = parse_coords(text, fmt)
    if len(q) < 2:
        return None
    return dequantize(q[0], width, bins), dequantize(q[1], height, bins)


# ---- masks -----------------------------------------------------------------------------------

def mask_to_rle(mask: np.ndarray) -> dict:
    """Binary mask (H, W) -> uncompressed COCO RLE {"size": [H, W], "counts": [...]}.

    COCO flattens in column-major (Fortran) order and the first count is the number of zeros,
    so a mask that starts with a foreground pixel begins with a 0 count.
    """
    flat = np.asarray(mask, dtype=bool).flatten(order="F")
    counts, prev, run = [], False, 0
    for v in flat:
        if v != prev:
            counts.append(run)
            prev, run = v, 0
        run += 1
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in rle["counts"]:
        flat[pos:pos + c] = val
        pos += c
        val = not val
    return flat.reshape((h, w), order="F")


def mask_to_box(mask: np.ndarray) -> Box | None:
    """Tightest pixel box around the foreground, with exclusive right/bottom edges."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def encode_polygon(points: list[tuple[float, float]], width: float, height: float,
                   fmt: str = "tokens", bins: int = LOC_BINS) -> str:
    """A mask outline as a flat list of quantised (x, y) vertices: one way to put a mask in text."""
    q: list[int] = []
    for x, y in points:
        q += [quantize(x, width, bins), quantize(y, height, bins)]
    return _format(q, fmt)


def decode_polygon(text: str, width: float, height: float, fmt: str = "tokens",
                   bins: int = LOC_BINS) -> list[tuple[float, float]] | None:
    """Grounding string -> polygon vertices at bin centres, or None when there are fewer than 3 vertices."""
    q = parse_coords(text, fmt)
    if len(q) < 6:
        return None
    q = q[:len(q) // 2 * 2]
    return [(dequantize(q[i], width, bins), dequantize(q[i + 1], height, bins)) for i in range(0, len(q), 2)]


def polygon_to_mask(points: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    """Rasterise a polygon to a binary mask (H, W), so a generated outline can be scored with mask IoU."""
    m = Image.new("L", (width, height), 0)
    ImageDraw.Draw(m).polygon(points, fill=255)
    return np.array(m) > 0
