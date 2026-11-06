"""How many vision tokens a page costs under three input schemes, and how many pixels a glyph gets.

Section "Resolution and the token budget" and Figure 4 of the post.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

A4_INCHES = (8.27, 11.69)

# InternVL's tiling rule is already built in tutorial C7 (c07-vlm/tinyvlm/image_processing.py). Reuse it.
_C07 = Path(__file__).resolve().parents[3] / "c07-vlm"
if _C07.exists() and str(_C07) not in sys.path:
    sys.path.append(str(_C07))


def page_pixels(dpi: float, inches=A4_INCHES) -> tuple[float, float]:
    return inches[0] * dpi, inches[1] * dpi


def post_formula_tokens(dpi: float) -> float:
    """The post's estimate: (8.27 d / 28) x (11.69 d / 28), about 0.123 d^2."""
    w, h = page_pixels(dpi)
    return (w / 28) * (h / 28)


def smart_resize(h: float, w: float, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 12845056):
    """Qwen2-VL's rule: round each side to a multiple of 28, then rescale if the area is outside [min, max].
    The default max_pixels is the one in Qwen2-VL's preprocessor_config.json (12,845,056 = 16,384 tokens)."""
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar = math.floor(h / beta / factor) * factor
        w_bar = math.floor(w / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return int(h_bar), int(w_bar)


def native_tokens(dpi: float, max_pixels: int = 12845056) -> int:
    """Native resolution (Qwen2-VL / Qwen2.5-VL): 14-pixel patches, 2 x 2 merged, so one token per 28 x 28 pixels."""
    w, h = page_pixels(dpi)
    hb, wb = smart_resize(round(h), round(w), max_pixels=max_pixels)
    return (hb // 28) * (wb // 28)


def tiling_tokens_figure(dpi: float, tile: int = 448, per_tile: int = 256, cap: int = 40) -> tuple[int, int]:
    """Figure 4's count: enough 448-pixel tiles to cover the page at its own size (at most 40), plus a thumbnail."""
    w, h = page_pixels(dpi)
    tiles = min(cap, math.ceil(w / tile) * math.ceil(h / tile))
    return tiles, (tiles + 1) * per_tile


def tiling_tokens_internvl(dpi: float, tile: int = 448, per_tile: int = 256, max_tiles: int = 40) -> tuple[int, int]:
    """InternVL's own rule: pick the grid whose aspect ratio is closest to the page's, resize the page to it, and add
    a thumbnail when there is more than one tile. The grid depends on the aspect ratio, so dpi barely matters."""
    from tinyvlm.image_processing import select_grid

    w, h = page_pixels(dpi)
    cols, rows = select_grid(round(w), round(h), tile_size=tile, max_tiles=max_tiles)
    n = cols * rows
    return n, (n + (n > 1)) * per_tile


def deepseek_mode_tokens(side: int) -> int:
    """DeepSeek-OCR's native modes: 16-pixel SAM patches, then 16x compression. 512 -> 64, 640 -> 100,
    1024 -> 256, 1280 -> 400."""
    return (side // 16) ** 2 // 16


def px_per_em(points: float, dpi: float) -> float:
    """A font of `points` points at `dpi` dots per inch is points x dpi / 72 pixels per em."""
    return points * dpi / 72


def effective_dpi_fixed(side: int, inches=A4_INCHES) -> float:
    """The resolution left when the whole page is shrunk so that its long side fits a square input of `side` pixels."""
    return side / max(inches)


def effective_dpi_internvl(dpi: float, tile: int = 448, max_tiles: int = 40) -> float:
    from tinyvlm.image_processing import select_grid

    w, h = page_pixels(dpi)
    cols, rows = select_grid(round(w), round(h), tile_size=tile, max_tiles=max_tiles)
    return min(cols * tile / A4_INCHES[0], rows * tile / A4_INCHES[1])
