"""How many screen pixels one visual token covers, and what happens to small icons inside it.

p = sqrt(W * H / N): a W x H screen resized to N square tokens gives each token a p x p patch of the real screen.
"""

from __future__ import annotations

import math

import numpy as np


def pixels_per_token(w: int, h: int, n_tokens: int) -> float:
    return math.sqrt(w * h / n_tokens)


def icon_strip(n_icons: int = 16, icon: int = 24, gap: int = 0, height: int = 48, seed: int = 0) -> tuple[np.ndarray, list]:
    """A toolbar: n_icons distinct grey levels, each icon x icon pixels, on a white strip. Returns image and boxes."""
    rng = np.random.default_rng(seed)
    w = n_icons * (icon + gap)
    img = np.full((height, w), 255.0)
    levels = rng.permutation(np.linspace(20, 200, n_icons))
    boxes = []
    y0 = (height - icon) // 2
    for i in range(n_icons):
        x0 = i * (icon + gap)
        img[y0:y0 + icon, x0:x0 + icon] = levels[i]
        boxes.append((x0, x0 + icon))
    return img, boxes


def icons_per_token(boxes: list, p: float) -> list[int]:
    """For each token column of width p, how many icons have their centre inside it."""
    width = boxes[-1][1]
    n_cols = math.ceil(width / p)
    counts = [0] * n_cols
    for x0, x1 in boxes:
        counts[min(int(((x0 + x1) / 2) // p), n_cols - 1)] += 1
    return counts


def distinguishable_icons(boxes: list, p: float) -> int:
    """Average-pool the strip into p-wide token columns. An icon stays separable only if it owns a token column
    alone: then that token's value is set by this icon and the background, not by a mix of neighbours."""
    counts = icons_per_token(boxes, p)
    return sum(1 for c in counts if c == 1)

