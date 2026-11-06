"""Synthetic scenes: coloured shapes on a grey canvas, with exact object masks on the image-token grid.

A scene is a list of shapes. Rendering gives a PIL image. `token_mask` says which image tokens an object covers,
so experiments can address "the circle's tokens" directly. Minimal pairs change one fact and keep every pixel
outside the edited object identical.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw

SIZE = 512
BACKGROUND = (235, 235, 235)
COLOURS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 160, 40), "yellow": (240, 200, 20)}
SHAPES = ("circle", "square", "triangle")


@dataclass(frozen=True)
class Shape:
    kind: str      # circle, square or triangle
    colour: str    # a key of COLOURS
    x: int         # left edge in pixels
    y: int         # top edge in pixels
    size: int      # side of the bounding box in pixels

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.size, self.y + self.size)


def _draw(draw: ImageDraw.ImageDraw, s: Shape, fill) -> None:
    x0, y0, x1, y1 = s.box
    if s.kind == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=fill)
    elif s.kind == "square":
        draw.rectangle([x0, y0, x1, y1], fill=fill)
    elif s.kind == "triangle":
        draw.polygon([((x0 + x1) // 2, y0), (x1, y1), (x0, y1)], fill=fill)
    else:
        raise ValueError(s.kind)


def render(shapes: list[Shape], outline_only: bool = False) -> Image.Image:
    im = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    d = ImageDraw.Draw(im)
    for s in shapes:
        if outline_only:  # line-drawing style, used for the overlap task
            x0, y0, x1, y1 = s.box
            d.ellipse([x0, y0, x1, y1], outline=COLOURS[s.colour], width=6)
        else:
            _draw(d, s, COLOURS[s.colour])
    return im


def blank() -> Image.Image:
    """The 'no information' image used by the blind baseline."""
    return Image.new("RGB", (SIZE, SIZE), BACKGROUND)


def pixel_mask(s: Shape) -> np.ndarray:
    im = Image.new("L", (SIZE, SIZE), 0)
    _draw(ImageDraw.Draw(im), s, 255)
    return np.asarray(im) > 0


def token_mask(s: Shape, grid: int = 8, min_cover: float = 0.1) -> np.ndarray:
    """Boolean [grid*grid] in raster order: True where the shape covers at least `min_cover` of the cell."""
    m = pixel_mask(s)
    cell = SIZE // grid
    cover = m.reshape(grid, cell, grid, cell).mean(axis=(1, 3))
    return (cover >= min_cover).reshape(-1)


def dilate(mask: np.ndarray, grid: int = 8) -> np.ndarray:
    """Add a one-token border around a token mask (Neo et al. ablate an object's tokens plus a border)."""
    m = mask.reshape(grid, grid)
    out = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1) & _no_wrap(dy, dx, grid)
    return out.reshape(-1)


def _no_wrap(dy: int, dx: int, grid: int) -> np.ndarray:
    keep = np.ones((grid, grid), bool)
    if dy == 1:
        keep[0, :] = False
    if dy == -1:
        keep[-1, :] = False
    if dx == 1:
        keep[:, 0] = False
    if dx == -1:
        keep[:, -1] = False
    return keep


def _overlaps(a: Shape, b: Shape, gap: int = 24) -> bool:
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    return not (ax1 + gap <= bx0 or bx1 + gap <= ax0 or ay1 + gap <= by0 or by1 + gap <= ay0)


def random_scene(rng: random.Random, n: int = 2, size_range=(130, 190)) -> list[Shape]:
    """n shapes with distinct kinds and distinct colours, not touching, so each question has one answer."""
    kinds = rng.sample(SHAPES, n)
    colours = rng.sample(list(COLOURS), n)
    shapes: list[Shape] = []
    while len(shapes) < n:
        size = rng.randint(*size_range)
        s = Shape(kinds[len(shapes)], colours[len(shapes)], rng.randint(16, SIZE - size - 16),
                  rng.randint(16, SIZE - size - 16), size)
        if all(not _overlaps(s, t) for t in shapes):
            shapes.append(s)
    return shapes


def recolour(shapes: list[Shape], index: int, colour: str) -> list[Shape]:
    """Minimal pair: the same scene with one shape's colour changed."""
    out = list(shapes)
    out[index] = replace(out[index], colour=colour)
    return out


def count_scene(rng: random.Random, n: int, size: int = 90) -> list[Shape]:
    """n same-colour circles that do not touch (the counting task)."""
    colour = rng.choice(list(COLOURS))
    shapes: list[Shape] = []
    while len(shapes) < n:
        s = Shape("circle", colour, rng.randint(16, SIZE - size - 16), rng.randint(16, SIZE - size - 16), size)
        if all(not _overlaps(s, t, gap=12) for t in shapes):
            shapes.append(s)
    return shapes


def overlap_scene(rng: random.Random, overlap: bool) -> list[Shape]:
    """Two outlined circles, either just overlapping or just apart (BlindTest's hardest regime: close shapes)."""
    r = rng.randint(70, 100)
    cx, cy = rng.randint(r + 20, SIZE // 2 - 10), rng.randint(r + 20, SIZE - r - 20)
    # centre distance: overlap if below 2r. Keep both cases close to the boundary.
    dist = 2 * r * (rng.uniform(0.70, 0.92) if overlap else rng.uniform(1.08, 1.30))
    ang = rng.uniform(-0.6, 0.6)
    cx2, cy2 = cx + dist * np.cos(ang), cy + dist * np.sin(ang)
    cx2, cy2 = int(min(cx2, SIZE - r - 4)), int(min(max(cy2, r + 4), SIZE - r - 4))
    c = rng.choice(list(COLOURS))
    a = Shape("circle", c, cx - r, cy - r, 2 * r)
    b = Shape("circle", c, cx2 - r, cy2 - r, 2 * r)
    return [a, b]


def circles_overlap(shapes: list[Shape]) -> bool:
    a, b = shapes
    r1, r2 = a.size / 2, b.size / 2
    d = np.hypot((a.x + r1) - (b.x + r2), (a.y + r1) - (b.y + r2))
    return bool(d < r1 + r2)
