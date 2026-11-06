"""Synthetic images, so every experiment runs offline and in seconds.

Two datasets:

- The "cat on a sofa" images of the post's regression example. One caption, two modes: the cat (a bright 2 x 3
  block) sits in the left half or the right half of an 8 x 8 image, at one of three heights.
- A shapes dataset of 16 x 16 images. Each image has one shape (square, disk, plus or frame) of random size,
  position and brightness. A caption names the shape and the side it is on, for example "disk left".
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------- the two-mode "cat" images

CAT_SIZE = 8


def cat_images(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """n images of shape (n, 8, 8). Half have the cat on the left, half on the right."""
    side = torch.randint(0, 2, (n,), generator=generator)
    row = torch.randint(2, 5, (n,), generator=generator)
    x = torch.zeros(n, CAT_SIZE, CAT_SIZE)
    for i in range(n):
        c0 = 0 if side[i] == 0 else CAT_SIZE - 3
        x[i, row[i] : row[i] + 2, c0 : c0 + 3] = 1.0
    return x


def side_balance(x: torch.Tensor) -> torch.Tensor:
    """Share of the image's brightness that is in the left half, per image."""
    half = x.shape[-1] // 2
    left = x[..., :half].clamp(min=0).sum(dim=(-1, -2))
    right = x[..., half:].clamp(min=0).sum(dim=(-1, -2))
    return left / (left + right + 1e-8)


# ---------------------------------------------------------------- shapes

SHAPES = ["square", "disk", "plus", "frame"]
SIDES = ["left", "right"]
IMG = 16


def _shape_mask(kind: int, s: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
    c = (s - 1) / 2
    if kind == 0:  # square
        return torch.ones(s, s)
    if kind == 1:  # disk
        return (((yy - c) ** 2 + (xx - c) ** 2) <= (s / 2) ** 2 * 0.85).float()
    if kind == 2:  # plus
        w = max(1, s // 3)
        lo = (s - w) // 2
        m = torch.zeros(s, s)
        m[lo : lo + w, :] = 1
        m[:, lo : lo + w] = 1
        return m
    m = torch.ones(s, s)  # frame: a hollow square
    m[1:-1, 1:-1] = 0
    return m


def shapes(n: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Return images (n, 1, 16, 16) in [0, 1], shape labels, side labels and pixel offsets."""
    g = torch.Generator().manual_seed(seed)
    kind = torch.randint(0, len(SHAPES), (n,), generator=g)
    size = torch.randint(6, 9, (n,), generator=g)
    bright = 0.55 + 0.45 * torch.rand(n, generator=g)
    images = torch.zeros(n, 1, IMG, IMG)
    ox = torch.zeros(n, dtype=torch.long)
    oy = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        s = int(size[i])
        ox[i] = torch.randint(0, IMG - s + 1, (1,), generator=g)
        oy[i] = torch.randint(0, IMG - s + 1, (1,), generator=g)
        images[i, 0, oy[i] : oy[i] + s, ox[i] : ox[i] + s] = bright[i] * _shape_mask(int(kind[i]), s)
    centre_x = ox + size.float() / 2
    side = (centre_x >= IMG / 2).long()
    return {"images": images, "shape": kind, "side": side, "x": ox, "y": oy, "size": size, "bright": bright}


# ---------------------------------------------------------------- caption vocabulary for the hybrid model

PAD, BOS, BOI, EOI = 0, 1, 2, 3
WORDS = ["<pad>", "<bos>", "<boi>", "<eoi>", *SHAPES, *SIDES]
SHAPE_TOKEN = {name: 4 + i for i, name in enumerate(SHAPES)}
SIDE_TOKEN = {name: 4 + len(SHAPES) + i for i, name in enumerate(SIDES)}


def caption_tokens(shape: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """(n, 2) token ids: the shape word, then the side word."""
    return torch.stack([4 + shape, 4 + len(SHAPES) + side], dim=1)


def to_patches(images: torch.Tensor, p: int = 4) -> torch.Tensor:
    """(n, 1, H, W) -> (n, (H/p)*(W/p), p*p), row-major patches."""
    n, _, h, w = images.shape
    x = images.reshape(n, h // p, p, w // p, p).permute(0, 1, 3, 2, 4)
    return x.reshape(n, (h // p) * (w // p), p * p)


def from_patches(patches: torch.Tensor, p: int = 4, h: int = IMG, w: int = IMG) -> torch.Tensor:
    n = patches.shape[0]
    x = patches.reshape(n, h // p, w // p, p, p).permute(0, 1, 3, 2, 4)
    return x.reshape(n, 1, h, w)
