"""How the image gets in: token counts for the four input strategies of the post's first question.

Each function returns a dict with the visual-token count and enough detail to see where it came
from. They are written from the papers' descriptions and checked in the tests against the
Hugging Face implementations where one exists (Qwen2-VL resizing, LLaVA-NeXT resolution choice
and feature count, InternVL / GOT-OCR2 tile grids).
"""

from __future__ import annotations

import math


# --- 1. fixed square --------------------------------------------------------------------------
def fixed_square(side: int = 336, patch: int = 14, merge: int = 1) -> dict:
    """Squash any image to side x side. LLaVA-1.5: 336 / 14 = 24, 24^2 = 576 tokens."""
    n = (side // patch) ** 2 // (merge * merge)
    return {"strategy": "fixed", "tiles": 1, "tokens": n}


# --- 2a. LLaVA-NeXT "anyres" tiling ------------------------------------------------------------
def select_best_resolution(orig_hw: tuple[int, int], candidates: list[tuple[int, int]]) -> tuple[int, int]:
    """Pick the canvas that keeps the most original pixels, then wastes the fewest (LLaVA-NeXT)."""
    oh, ow = orig_hw
    best, best_eff, best_waste = None, 0, float("inf")
    for h, w in candidates:
        scale = min(w / ow, h / oh)
        eff = min(int(ow * scale) * int(oh * scale), ow * oh)
        waste = h * w - eff
        if eff > best_eff or (eff == best_eff and waste < best_waste):
            best, best_eff, best_waste = (h, w), eff, waste
    return best


def llava_next_tokens(orig_hw: tuple[int, int], pinpoints: list[tuple[int, int]], tile: int = 336,
                      patch: int = 14, newline: bool = True) -> dict:
    """LLaVA-NeXT as served by HF: a 576-token global view, plus the tiled canvas with the padding
    rows or columns removed ("unpadded") and one newline token per feature row."""
    oh, ow = orig_hw
    ch, cw = select_best_resolution(orig_hw, pinpoints)
    side = tile // patch
    cur_h, cur_w = side * (ch // tile), side * (cw // tile)
    if ow / oh > cur_w / cur_h:          # wider than the canvas: padding rows top and bottom
        new_h = int(round(oh * (cur_w / ow), 7))
        cur_h -= 2 * ((cur_h - new_h) // 2)
    else:                                 # taller: padding columns left and right
        new_w = int(round(ow * (cur_h / oh), 7))
        cur_w -= 2 * ((cur_w - new_w) // 2)
    tiled = cur_h * cur_w + (cur_h if newline else 0)
    return {"strategy": "llava-next tiles", "canvas": (ch, cw), "tiles": (ch // tile) * (cw // tile) + 1,
            "tokens": side * side + tiled}


# --- 2b. InternVL dynamic tiling ---------------------------------------------------------------
def tile_grids(min_tiles: int, max_tiles: int) -> list[tuple[int, int]]:
    """All (columns, rows) with min_tiles <= columns * rows <= max_tiles, fewest tiles first."""
    grids = [(c, r) for c in range(1, max_tiles + 1) for r in range(1, max_tiles + 1)
             if min_tiles <= c * r <= max_tiles]
    return sorted(grids, key=lambda x: x[0] * x[1])


def closest_grid(orig_hw: tuple[int, int], tile: int = 448, min_tiles: int = 1, max_tiles: int = 12) -> tuple[int, int]:
    """InternVL's rule: the grid whose aspect ratio is closest to the image's. On a tie, take the
    grid with more tiles while the image area is above half the tiled area."""
    oh, ow = orig_hw
    ar, area = ow / oh, ow * oh
    best, best_diff = (1, 1), float("inf")
    for c, r in tile_grids(min_tiles, max_tiles):
        diff = abs(ar - c / r)
        if diff < best_diff:
            best, best_diff = (c, r), diff
        elif diff == best_diff and area > 0.5 * tile * tile * c * r:
            best = (c, r)
    return best


def internvl_tokens(orig_hw: tuple[int, int], tile: int = 448, patch: int = 14, shuffle: int = 2,
                    max_tiles: int = 12, thumbnail: bool = True) -> dict:
    """Tiles of 448^2, each 1024 patches, pixel-shuffled to 256 tokens, plus a thumbnail tile."""
    c, r = closest_grid(orig_hw, tile, 1, max_tiles)
    n_tiles = c * r + (1 if thumbnail and c * r > 1 else 0)
    per_tile = (tile // patch) ** 2 // (shuffle * shuffle)
    return {"strategy": "internvl tiles", "grid": (c, r), "tiles": n_tiles, "tokens": n_tiles * per_tile}


# --- 3. native resolution ----------------------------------------------------------------------
def smart_resize(h: int, w: int, factor: int = 28, min_pixels: int = 56 * 56,
                 max_pixels: int = 14 * 14 * 4 * 1280) -> tuple[int, int]:
    """Qwen2-VL: round both sides to a multiple of `factor` (patch 14 x merge 2), keeping the
    pixel count inside [min_pixels, max_pixels] and the aspect ratio about the same."""
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def native_tokens(orig_hw: tuple[int, int], patch: int = 14, merge: int = 2, min_pixels: int = 3136,
                  max_pixels: int = 12_845_056) -> dict:
    """Qwen2-VL style: the patch grid follows the image, then a merge x merge merge. The pixel
    limits default to Qwen2-VL-2B-Instruct's preprocessor_config.json (4 to 16 384 tokens)."""
    h, w = smart_resize(*orig_hw, factor=patch * merge, min_pixels=min_pixels, max_pixels=max_pixels)
    return {"strategy": "native", "resized": (h, w), "tiles": 1, "tokens": (h // patch) * (w // patch) // merge ** 2}


# --- 4. overlapping multi-crop -----------------------------------------------------------------
def overlapping_crops(orig_hw: tuple[int, int], crop: int = 336, overlap: int = 56) -> tuple[int, int]:
    """Number of (columns, rows) of fixed-size windows with `overlap` pixels shared between
    neighbours, so no detail sits only on a crop boundary."""
    stride = crop - overlap
    oh, ow = orig_hw
    cols = max(1, math.ceil((ow - overlap) / stride))
    rows = max(1, math.ceil((oh - overlap) / stride))
    return cols, rows


def multicrop_tokens(orig_hw: tuple[int, int], crop: int = 336, overlap: int = 56, tokens_per_crop: int = 144,
                     global_view: bool = True, max_side: int = 1008) -> dict:
    """Overlapping crops plus an optional global view. The image is first shrunk so its longer
    side is at most `max_side` (no upscaling). The overlap is paid for in tokens."""
    scale = min(1.0, max_side / max(orig_hw))
    orig_hw = (round(orig_hw[0] * scale), round(orig_hw[1] * scale))
    c, r = overlapping_crops(orig_hw, crop, overlap)
    n = c * r + (1 if global_view else 0)
    stride = crop - overlap
    covered = ((c - 1) * stride + crop) * ((r - 1) * stride + crop)
    return {"strategy": "multi-crop", "grid": (c, r), "tiles": n, "tokens": n * tokens_per_crop,
            "pixels_encoded_over_covered": round(c * r * crop * crop / covered, 3)}
