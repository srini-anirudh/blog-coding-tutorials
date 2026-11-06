"""Visual tokens per image under each family's preprocessing rule, written from scratch.

Three kinds of rule:

    fixed    resize to one square; the token count never changes      LLaVA-1.5, Qwen-VL, Gemma 3
    tiling   cut into encoder-sized tiles plus a thumbnail             LLaVA-NeXT, OneVision, InternVL, Idefics3
    native   one token per merged patch of the image at its own size   Qwen2-VL / 2.5-VL, Pixtral

Each function returns a `Count`: the number of tokens that stand for the image in the language model's sequence,
and the grid that produced it. `scripts/02_check_processors.py` checks these rules against the real Hugging Face
processors. All sizes are (width, height) in pixels unless a name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Count:
    tokens: int  # tokens in the LLM sequence that carry the image
    kind: str  # fixed | tiling | native
    detail: str = ""
    grid: tuple[int, int] = (1, 1)  # (columns, rows) of tiles, or of tokens for native rules
    parts: dict = field(default_factory=dict)  # the terms that add up to `tokens`


# ----------------------------------------------------------------------------- fixed


def fixed(side: int, patch: int, pool: int = 1, extra: int = 0, name: str = "") -> Count:
    """Resize to side x side, one token per (patch * pool) square, plus `extra` tokens (for example a CLS)."""
    per_side = side // patch // pool
    return Count(per_side * per_side + extra, "fixed", f"{name} {side} px square, {per_side} x {per_side} tokens",
                 (per_side, per_side))


def llava15(width: int, height: int) -> Count:
    return fixed(336, 14, name="LLaVA-1.5:")  # 24 x 24 = 576, CLS dropped


def qwen_vl(width: int, height: int) -> Count:
    return Count(256, "fixed", "Qwen-VL: 448 px square, 1,024 patches read by 256 learned queries")


def gemma3(width: int, height: int) -> Count:
    return fixed(896, 14, pool=4, name="Gemma 3:")  # 64 x 64 patches, 4 x 4 average pool -> 16 x 16 = 256


# ----------------------------------------------------------------------------- native


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56,
                 max_pixels: int = 16384 * 28 * 28) -> tuple[int, int]:
    """Qwen2-VL's resize: round both sides to multiples of `factor`, then rescale into [min_pixels, max_pixels]."""
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


def qwen2_vl(width: int, height: int, patch: int = 14, merge: int = 2,
             min_pixels: int = 56 * 56, max_pixels: int = 16384 * 28 * 28) -> Count:
    """Native resolution: 14 px patches, then a 2 x 2 merge, so one token per 28 px square.

    The pixel limits are the HF processor defaults: 4 to 16,384 merged tokens.
    """
    f = patch * merge
    h, w = smart_resize(height, width, f, min_pixels, max_pixels)
    cols, rows = w // f, h // f
    return Count(cols * rows, "native", f"Qwen2-VL: resized to {w} x {h}, {cols} x {rows} merged tokens",
                 (cols, rows), {"grid_thw": (1, h // patch, w // patch)})


def pixtral(width: int, height: int, patch: int = 16, longest_edge: int = 1024) -> Count:
    """Native resolution, capped at `longest_edge`; a [IMG_BREAK] after each row and [IMG_END] after the last."""
    ratio = max(height / longest_edge, width / longest_edge)
    if ratio > 1:
        height, width = int(math.floor(height / ratio)), int(math.floor(width / ratio))
    rows, cols = (height - 1) // patch + 1, (width - 1) // patch + 1
    return Count(rows * (cols + 1), "native", f"Pixtral: {cols} x {rows} patches + {rows} row-end tokens",
                 (cols, rows), {"patches": rows * cols, "row_breaks": rows})


# ----------------------------------------------------------------------------- tiling


def select_best_resolution(width: int, height: int, candidates: list[tuple[int, int]]) -> tuple[int, int]:
    """LLaVA-NeXT's rule: the canvas (w, h) that keeps the most original pixels, then wastes the fewest."""
    best, best_eff, best_waste = None, -1, float("inf")
    for cw, ch in candidates:
        scale = min(cw / width, ch / height)
        eff = min(int(width * scale) * int(height * scale), width * height)
        waste = cw * ch - eff
        if eff > best_eff or (eff == best_eff and waste < best_waste):
            best, best_eff, best_waste = (cw, ch), eff, waste
    return best


def _unpadded(width: int, height: int, grid_w: int, grid_h: int) -> tuple[int, int]:
    """Token rows and columns left after LLaVA-NeXT removes the padding added to fit the canvas."""
    if width / height > grid_w / grid_h:
        new_h = int(round(height * (grid_w / width), 7))
        grid_h -= 2 * ((grid_h - new_h) // 2)
    else:
        new_w = int(round(width * (grid_h / height), 7))
        grid_w -= 2 * ((grid_w - new_w) // 2)
    return grid_w, grid_h


LLAVA_NEXT_PINPOINTS = [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]  # (w, h)


def llava_next(width: int, height: int, tile: int = 336, patch: int = 14,
               pinpoints: list[tuple[int, int]] = LLAVA_NEXT_PINPOINTS) -> Count:
    """AnyRes: a base image plus a grid of tiles. Padding tokens are removed and one newline token ends each row."""
    side = tile // patch  # 24 tokens per tile side
    cw, ch = select_best_resolution(width, height, pinpoints)
    cols, rows = cw // tile, ch // tile
    gw, gh = _unpadded(width, height, cols * side, rows * side)
    base = side * side
    total = base + gw * gh + gh
    return Count(total, "tiling", f"LLaVA-NeXT: {cols} x {rows} tiles of {tile} px + base; "
                 f"{gw} x {gh} unpadded tokens + {gh} newlines", (cols, rows),
                 {"base": base, "tiles": gw * gh, "newlines": gh, "naive": base * (cols * rows + 1)})


def onevision_pinpoints(tile: int = 384, max_grid: int = 6) -> list[tuple[int, int]]:
    return [(tile * c, tile * r) for r in range(1, max_grid + 1) for c in range(1, max_grid + 1)]


def llava_onevision(width: int, height: int, tile: int = 384, patch: int = 14, max_tiles: int = 9) -> Count:
    """AnyResMax-9: like AnyRes, but the tile tokens are shrunk by interpolation when they exceed 9 tiles' worth."""
    side = tile // patch  # 27
    cw, ch = select_best_resolution(width, height, onevision_pinpoints(tile))
    cols, rows = cw // tile, ch // tile
    gw, gh = _unpadded(width, height, cols * side, rows * side)
    ratio = math.sqrt(gw * gh / (max_tiles * side * side))
    if ratio > 1.1:
        gw, gh = int(gw // ratio), int(gh // ratio)
    base = side * side
    return Count(base + gw * gh + gh, "tiling", f"OneVision: {cols} x {rows} canvas, tile tokens {gw} x {gh} "
                 f"after unpadding{' and interpolation' if ratio > 1.1 else ''}", (cols, rows),
                 {"base": base, "tiles": gw * gh, "newlines": gh})


def tile_grids(min_tiles: int = 1, max_tiles: int = 12) -> list[tuple[int, int]]:
    """Every (columns, rows) with min_tiles <= columns * rows <= max_tiles, fewest tiles first."""
    grids = {(c, r) for n in range(min_tiles, max_tiles + 1) for c in range(1, n + 1) for r in range(1, n + 1)
             if min_tiles <= c * r <= max_tiles}
    return sorted(grids, key=lambda g: g[0] * g[1])


def internvl_grid(width: int, height: int, tile: int = 448, max_tiles: int = 12) -> tuple[int, int]:
    """InternVL 1.5's rule: the grid whose aspect ratio is closest to the image's.

    On a tie, the larger grid wins while the image area exceeds half the grid's area.
    """
    aspect, area = width / height, width * height
    best, best_diff = (1, 1), float("inf")
    for c, r in tile_grids(1, max_tiles):
        diff = abs(aspect - c / r)
        if diff < best_diff:
            best, best_diff = (c, r), diff
        elif diff == best_diff and area > 0.5 * tile * tile * c * r:
            best = (c, r)
    return best


def internvl(width: int, height: int, tile: int = 448, tokens_per_tile: int = 256, max_tiles: int = 12) -> Count:
    """Aspect-ratio tiling: 448 px tiles, pixel shuffle to 256 tokens each, plus a thumbnail if there are 2+ tiles."""
    c, r = internvl_grid(width, height, tile, max_tiles)
    n = c * r + (1 if c * r > 1 else 0)
    return Count(n * tokens_per_tile, "tiling", f"InternVL: {c} x {r} tiles{' + thumbnail' if c * r > 1 else ''}, "
                 f"{tokens_per_tile} tokens each", (c, r), {"tiles": n})


def idefics3(width: int, height: int, longest_edge: int = 4 * 364, tile: int = 364,
             tokens_per_tile: int = 169) -> Count:
    """Idefics3 / SmolVLM: scale the long side to `longest_edge`, round both sides up to whole tiles, split.

    A global view is added when there is more than one tile. Tags such as <row_1_col_1> sit between tiles; they are
    ordinary text tokens and are not counted here.
    """
    aspect = width / height
    if width >= height:  # step 1: long side to longest_edge, other side kept even
        w, h = longest_edge, int(longest_edge / aspect)
        h += h % 2
    else:
        h, w = longest_edge, int(longest_edge * aspect)
        w += w % 2
    w, h = min(w, 4096), min(h, 4096)
    aspect = w / h  # step 2: round up to multiples of the tile
    if w >= h:
        w = math.ceil(w / tile) * tile
        h = math.ceil(int(w / aspect) / tile) * tile
    else:
        h = math.ceil(h / tile) * tile
        w = math.ceil(int(h * aspect) / tile) * tile
    cols, rows = w // tile, h // tile
    n = cols * rows + 1 if cols * rows > 1 else 1
    return Count(n * tokens_per_tile, "tiling", f"Idefics3: {cols} x {rows} tiles of {tile} px"
                 f"{' + global view' if n > 1 else ''}, {tokens_per_tile} tokens each", (cols, rows), {"tiles": n})


def deepseek_vl2(width: int, height: int, tile: int = 384, max_tiles: int = 9) -> Count:
    """DeepSeek-VL2: best m x n grid (m * n <= 9) of 384 px tiles plus a thumbnail; 2 x 2 pixel shuffle to 14 x 14.

    Each row of tokens ends with a newline token, and one separator splits the thumbnail from the tiles. This is
    the rule as the paper describes it; the processor needs remote code, so it is not checked here.
    """
    cands = [(c * tile, r * tile) for r in range(1, max_tiles + 1) for c in range(1, max_tiles + 1)
             if c * r <= max_tiles]
    cw, ch = select_best_resolution(width, height, cands)
    cols, rows = cw // tile, ch // tile
    side = 14
    thumb = side * (side + 1)
    tiles = rows * side * (cols * side + 1)
    return Count(thumb + 1 + tiles, "tiling", f"DeepSeek-VL2: {cols} x {rows} tiles + thumbnail", (cols, rows))


def minicpm_v25(width: int, height: int, tokens_per_slice: int = 96, max_slices: int = 9,
                encoder_side: int = 448) -> Count:
    """LLaVA-UHD slicing as the post describes it: slices follow image area, at most 9, plus the whole image."""
    n = min(max_slices, math.ceil(width * height / (encoder_side * encoder_side)))
    tokens = tokens_per_slice if n <= 1 else tokens_per_slice * (n + 1)
    return Count(tokens, "tiling", f"MiniCPM-V 2.5: about {n} slices + the whole image, {tokens_per_slice} each")


RULES = {
    "LLaVA-1.5": llava15,
    "Qwen-VL": qwen_vl,
    "Gemma 3": gemma3,
    "MiniCPM-V 2.5": minicpm_v25,
    "Idefics3": idefics3,
    "DeepSeek-VL2": deepseek_vl2,
    "LLaVA-NeXT": llava_next,
    "LLaVA-OneVision": llava_onevision,
    "InternVL 1.5-3": internvl,
    "Pixtral": pixtral,
    "Qwen2/2.5-VL": qwen2_vl,
}

# The preset images of the post's Figure 6, (name, width, height).
PRESETS = [("Square photo", 1024, 1024), ("Phone photo", 4032, 3024), ("Letter scan", 1700, 2200),
           ("Receipt", 600, 2000), ("Slide", 1920, 1080), ("Small icon", 300, 300)]


# ----------------------------------------------------------------------------- the post's simpler formulas

def post_formula(name: str, width: int, height: int) -> int | None:
    """The simplified count Figure 6 of the post shows, for rules that it marks as approximations (*).

    LLaVA-NeXT: 576 per tile plus the base image, padding kept. Idefics3: long side capped at 1,820 px, 169 per tile.
    Pixtral: patches only, row breaks ignored. Everything else matches the exact rule.
    """
    if name == "LLaVA-NeXT":
        return llava_next(width, height).parts["naive"]
    if name == "Idefics3":
        s = min(1.0, 1820 / max(width, height))
        c, r = math.ceil(width * s / 364), math.ceil(height * s / 364)
        return 169 if c * r <= 1 else 169 * (c * r + 1)
    if name == "Pixtral":
        return pixtral(width, height).parts["patches"]
    return None
