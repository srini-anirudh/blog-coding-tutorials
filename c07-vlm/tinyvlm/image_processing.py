"""The image processor: PIL image -> crops -> normalised float tensor.

Two jobs live here.

1. Turn one picture into the exact tensor the vision encoder was trained on:
   RGB, resized to 224 x 224 with bicubic filtering, scaled to [0, 1], then normalised with the
   encoder's mean and std. For SigLIP the mean and std are both 0.5, so pixels land in [-1, 1].
2. Decide how a picture of arbitrary size becomes one or more 224 x 224 crops
   (fixed resize, aspect-ratio padding, or tiling). The number of crops sets the number of image tokens.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_base import ImageProcessingMixin

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
STRATEGIES = ("resize", "pad", "tile")


def to_rgb(image: Image.Image) -> Image.Image:
    """Palette, grayscale and RGBA images have 1, 2 or 4 channels; the encoder expects exactly 3."""
    return image if image.mode == "RGB" else image.convert("RGB")


def resize(image: Image.Image, size: int, resample: int = Image.Resampling.BICUBIC) -> Image.Image:
    """Squash to size x size, ignoring the aspect ratio. This is how SigLIP saw images in training."""
    return image.resize((size, size), resample=resample)


def normalize(image: Image.Image, mean=SIGLIP_MEAN, std=SIGLIP_STD) -> torch.Tensor:
    """uint8 H x W x 3 picture -> float32 tensor of shape 3 x H x W, (x / 255 - mean) / std."""
    array = np.asarray(image, dtype=np.float32) / 255.0  # H x W x 3, values in [0, 1]
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())  # channels first, as conv layers expect


def pad_to_square(image: Image.Image, mean=SIGLIP_MEAN) -> Image.Image:
    """Centre the picture on a square canvas filled with the mean colour.

    After normalisation the padding is close to 0, so it carries almost no signal. It is not exactly 0:
    a mean of 0.5 is 127.5 in 8 bits, which rounds to 128 and normalises to 0.0039.
    """
    width, height = image.size
    side = max(width, height)
    fill = tuple(round(m * 255) for m in mean)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def candidate_grids(max_tiles: int) -> list[tuple[int, int]]:
    """Every (cols, rows) grid with at most `max_tiles` tiles, smallest area first."""
    grids = [(c, r) for c in range(1, max_tiles + 1) for r in range(1, max_tiles + 1) if c * r <= max_tiles]
    return sorted(grids, key=lambda g: (g[0] * g[1], g[0]))


def select_grid(width: int, height: int, tile_size: int = 224, max_tiles: int = 6) -> tuple[int, int]:
    """Pick the tile grid whose aspect ratio is closest to the picture's (InternVL's rule).

    On a tie (2:1 and 4:2 are the same ratio) take the bigger grid, but only if the picture has
    enough pixels to fill more than half of it. Small pictures should not be blown up into many tiles.
    """
    aspect = width / height
    best, best_diff = (1, 1), math.inf
    for cols, rows in candidate_grids(max_tiles):
        diff = abs(aspect - cols / rows)
        if diff < best_diff:
            best, best_diff = (cols, rows), diff
        elif diff == best_diff and width * height > 0.5 * tile_size * tile_size * cols * rows:
            best = (cols, rows)
    return best


def tile(image: Image.Image, tile_size: int = 224, max_tiles: int = 6, thumbnail: bool = True) -> list[Image.Image]:
    """Resize to the chosen grid, cut it into tile_size squares (row-major), add a global thumbnail.

    The tiles keep local detail; the thumbnail keeps the whole scene in one view.
    """
    cols, rows = select_grid(*image.size, tile_size=tile_size, max_tiles=max_tiles)
    grid = image.resize((cols * tile_size, rows * tile_size), resample=Image.Resampling.BICUBIC)
    tiles = [
        grid.crop((c * tile_size, r * tile_size, (c + 1) * tile_size, (r + 1) * tile_size))
        for r in range(rows)
        for c in range(cols)
    ]
    if thumbnail and len(tiles) > 1:
        tiles.append(resize(image, tile_size))
    return tiles


class TinyVLMImageProcessor(ImageProcessingMixin):
    """Image processor with Hugging Face `save_pretrained` / `from_pretrained` (preprocessor_config.json).

    Output of `__call__` for a list of n pictures:
        pixel_values: float32, (total_crops, 3, size, size), crops of picture 0 first, then picture 1, ...
        num_crops:    list[int] of length n, crops per picture (1 unless strategy == "tile")
    """

    def __init__(
        self,
        size: int = 224,
        image_mean=SIGLIP_MEAN,
        image_std=SIGLIP_STD,
        resample: int = int(Image.Resampling.BICUBIC),
        strategy: str = "resize",
        max_tiles: int = 6,
        use_thumbnail: bool = True,
        **kwargs,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        super().__init__(**kwargs)
        self.size = size
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.resample = resample
        self.strategy = strategy
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail

    def crops(self, image: Image.Image) -> list[Image.Image]:
        """The resolution strategy: one picture in, a list of size x size crops out."""
        image = to_rgb(image)
        if self.strategy == "pad":
            image = pad_to_square(image, self.image_mean)
        if self.strategy == "tile":
            return tile(image, self.size, self.max_tiles, self.use_thumbnail)
        return [resize(image, self.size, self.resample)]

    def num_crops_for_size(self, width: int, height: int) -> int:
        """How many crops a width x height picture becomes, without touching any pixels."""
        if self.strategy != "tile":
            return 1
        cols, rows = select_grid(width, height, self.size, self.max_tiles)
        n = cols * rows
        return n + 1 if (self.use_thumbnail and n > 1) else n

    def __call__(self, images: list[Image.Image]) -> dict:
        crops_per_image = [self.crops(image) for image in images]
        pixels = [normalize(c, self.image_mean, self.image_std) for crops in crops_per_image for c in crops]
        return {
            "pixel_values": torch.stack(pixels) if pixels else torch.zeros(0, 3, self.size, self.size),
            "num_crops": [len(crops) for crops in crops_per_image],
        }
