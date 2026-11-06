"""A view is a box of the original photo plus the size the encoder receives it at.

Looking again means rendering a new view from the original pixels, never from an earlier view. Each rendered image
carries its view in `img.info`, which the stub reader in the tests uses in place of eyes. The real model ignores it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .scene import GRID, TEXT_H, H, W, Scene, cell_box, resize_to


@dataclass(frozen=True)
class View:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in original pixels
    enc_w: int
    enc_h: int

    @property
    def text_px(self) -> float:
        """Plaque digit height in this view: h_enc = h_orig * W_enc / W_region."""
        return TEXT_H * self.enc_w / (self.box[2] - self.box[0])

    def contains(self, box) -> bool:
        x0, y0, x1, y1 = self.box
        return box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1


def snap28(x: float) -> int:
    return max(28, round(x / 28) * 28)


def full_view(width: int) -> View:
    return View((0, 0, W, H), snap28(width), snap28(width * H / W))


def cell_view(col: int, row: int, grid: int = GRID, width: int = 448) -> View:
    x0, y0, x1, y1 = cell_box(col, row, grid)
    return View((x0, y0, x1, y1), width, snap28(width * (y1 - y0) / (x1 - x0)))


def random_view(rng: random.Random, min_frac=0.25, max_frac=0.5, width: int = 448) -> View:
    """A random 4:3 crop whose width is between min_frac and max_frac of the photo."""
    f = rng.uniform(min_frac, max_frac)
    w, h = int(W * f), int(H * f)
    x0, y0 = rng.randint(0, W - w), rng.randint(0, H - h)
    return View((x0, y0, x0 + w, y0 + h), width, snap28(width * h / w))


def render(scene: Scene, view: View) -> Image.Image:
    img = resize_to(scene.image.crop(view.box), view.enc_w, view.enc_h)
    img.info["view"] = view
    img.info["scene"] = scene
    return img


def detail_retained(scene: Scene, view: View) -> float:
    """How much of the plaque's fine detail survives a view: the correlation between horizontal intensity gradients
    of the original plaque and of the view scaled back up. 1 is perfect; near 0 means the letters are gone."""
    x0, y0, x1, y1 = scene.plaque_box
    bx, by = view.box[:2]
    up = render(scene, view).resize((view.box[2] - bx, view.box[3] - by), Image.BILINEAR)
    ref = np.asarray(scene.image.crop(scene.plaque_box).convert("L"), float)
    got = np.asarray(up.crop((x0 - bx, y0 - by, x1 - bx, y1 - by)).convert("L"), float)
    a, b = np.diff(ref, axis=1).ravel(), np.diff(got, axis=1).ravel()
    return float(np.corrcoef(a, b)[0, 1])
