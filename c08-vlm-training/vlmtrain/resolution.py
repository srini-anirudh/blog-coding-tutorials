"""Image preprocessing with three resolution strategies.

The vision encoder only sees square tiles of `tile` pixels (224 for SigLIP-base/16). What changes is
how an arbitrary image becomes tiles, and therefore how many visual tokens it costs:

    resize   squash the whole image to one tile                     1 tile, any aspect ratio distorted
    pad      letterbox to a square, then one tile                   1 tile, aspect kept, pixels wasted
    tile     pick a rows x cols grid that matches the aspect ratio,  rows*cols (+1 thumbnail) tiles
             cut full-resolution tiles, add a global thumbnail

Output per image: a tensor (n_tiles, 3, tile, tile), normalised the way SigLIP was trained
(mean 0.5, std 0.5 per channel, so pixels land in [-1, 1]).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

MEAN, STD = 0.5, 0.5


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB (H, W) -> float tensor (3, H, W) in [-1, 1]."""
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
    return ((x - MEAN) / STD).permute(2, 0, 1).contiguous()


def choose_grid(width: int, height: int, tile: int, max_tiles: int) -> tuple[int, int]:
    """(rows, cols) with rows*cols <= max_tiles whose aspect ratio is closest to the image's.

    Ties go to the grid whose pixel count is closest to the image's, so a small image is not blown
    up into many tiles just because the ratios match (InternVL uses the same tie-break idea).
    """
    best, best_key = (1, 1), None
    target = math.log(width / height)
    for rows in range(1, max_tiles + 1):
        for cols in range(1, max_tiles // rows + 1):
            ratio_err = abs(math.log(cols / rows) - target)
            area_err = abs(math.log(rows * cols * tile * tile / (width * height)))
            key = (round(ratio_err, 6), area_err)
            if best_key is None or key < best_key:
                best, best_key = (rows, cols), key
    return best


def tiles_for(img: Image.Image, strategy: str = "tile", tile: int = 224, max_tiles: int = 4) -> list[Image.Image]:
    """The list of tile images the encoder will see, in the order their tokens appear in the sequence."""
    if strategy == "resize":
        return [img.resize((tile, tile), Image.BICUBIC)]
    if strategy == "pad":
        side = max(img.size)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        return [canvas.resize((tile, tile), Image.BICUBIC)]
    if strategy != "tile":
        raise ValueError(f"unknown resolution strategy {strategy!r}")
    rows, cols = choose_grid(img.width, img.height, tile, max_tiles)
    if rows * cols == 1:
        return [img.resize((tile, tile), Image.BICUBIC)]
    big = img.resize((cols * tile, rows * tile), Image.BICUBIC)
    crops = [big.crop((c * tile, r * tile, (c + 1) * tile, (r + 1) * tile)) for r in range(rows) for c in range(cols)]
    # The thumbnail comes first: a low-resolution view of the whole layout, then the detail tiles.
    return [img.resize((tile, tile), Image.BICUBIC)] + crops


def preprocess(img: Image.Image, strategy: str = "tile", tile: int = 224, max_tiles: int = 4) -> torch.Tensor:
    """PIL image -> (n_tiles, 3, tile, tile)."""
    return torch.stack([to_tensor(t) for t in tiles_for(img, strategy, tile, max_tiles)])


def n_tiles(width: int, height: int, strategy: str = "tile", tile: int = 224, max_tiles: int = 4) -> int:
    """Tile count without touching pixels (used for token budgeting and for the figures)."""
    if strategy in ("resize", "pad"):
        return 1
    rows, cols = choose_grid(width, height, tile, max_tiles)
    return 1 if rows * cols == 1 else rows * cols + 1
