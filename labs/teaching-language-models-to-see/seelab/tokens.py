"""What a visual token costs: token counts under four resolution strategies, and which tokens carry the loss."""

from __future__ import annotations

import math


def square_tokens(side: int, patch: int, merge: int = 1) -> int:
    """Resize to side x side, cut into patch x patch squares, merge each merge x merge block into one token."""
    g = side // patch
    return (g // merge) ** 2


def ceil_grid_tokens(width: int, height: int, patch: int, merge: int = 1) -> tuple[int, int, int]:
    """The rule in the post's Figure 7: round the patch grid up to a multiple of `merge`."""
    gw = math.ceil(math.ceil(width / patch) / merge) * merge
    gh = math.ceil(math.ceil(height / patch) / merge) * merge
    return gw // merge, gh // merge, gw * gh // merge ** 2


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56,
                 max_pixels: int = 14 * 14 * 4 * 16384) -> tuple[int, int]:
    """Qwen2-VL's native-resolution resize: round each side to the nearest multiple of factor = patch x merge.

    Re-implemented from the Qwen2-VL image processor; `s5_token_budget.py` checks it against the real one.
    The defaults are that processor's limits: 3,136 to 12,845,056 pixels (4 to 16,384 tokens).
    """
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = math.floor(height / beta / factor) * factor
        w = math.floor(width / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def native_tokens(width: int, height: int, patch: int = 14, merge: int = 2) -> tuple[int, int, int]:
    h, w = smart_resize(height, width, factor=patch * merge)
    return w // (patch * merge), h // (patch * merge), (h // patch) * (w // patch) // merge ** 2


def internvl_grid(width: int, height: int, min_num: int = 1, max_num: int = 12, tile: int = 448) -> tuple[int, int]:
    """InternVL 1.5's dynamic tiling: the (columns, rows) grid whose aspect ratio is closest to the image's.

    Re-implemented from InternVL's public preprocessing code. Ties go to the larger grid when the
    image has enough pixels to fill it.
    """
    aspect = width / height
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}, key=lambda r: r[0] * r[1])
    best, best_diff = (1, 1), float("inf")
    for i, j in ratios:
        diff = abs(aspect - i / j)
        if diff < best_diff:
            best, best_diff = (i, j), diff
        elif diff == best_diff and width * height > 0.5 * tile * tile * i * j:
            best = (i, j)
    return best


def tiled_tokens(width: int, height: int, max_num: int, tile: int = 448, patch: int = 14, merge: int = 2,
                 thumbnail: bool = True) -> tuple[int, int, int]:
    """Number of tiles (plus a thumbnail when there is more than one) times the tokens of one tile."""
    cols, rows = internvl_grid(width, height, max_num=max_num, tile=tile)
    n = cols * rows + (1 if thumbnail and cols * rows > 1 else 0)
    return cols, rows, n * square_tokens(tile, patch, merge)


def loss_share(segments: list[tuple[str, int, bool]]) -> tuple[int, int, float]:
    """segments: (kind, length, carries_loss). Returns (scored tokens, total tokens, share)."""
    total = sum(n for _, n, _ in segments)
    scored = sum(n for _, n, s in segments if s)
    return scored, total, scored / total


def train_flops(params: float, tokens: int) -> float:
    """About 6 FLOPs per parameter per token for one training step (forward 2, backward 4)."""
    return 6 * params * tokens


def adam_bytes(trained: float, frozen: float) -> float:
    """Mixed-precision Adam rule of thumb: 16 bytes per trained parameter, 2 per frozen (bf16 weights only)."""
    return 16 * trained + 2 * frozen
