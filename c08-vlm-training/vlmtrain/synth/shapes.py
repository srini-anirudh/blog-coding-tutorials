"""Rendered shape scenes: coloured circles, squares and triangles with exact geometry.

Every scene returns the image and the list of objects that were drawn, so captions, counts,
colours, boxes and masks are all known exactly. No label is ever estimated.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (215, 40, 40),
    "green": (40, 155, 65),
    "blue": (45, 85, 215),
    "yellow": (225, 195, 25),
    "purple": (135, 60, 175),
    "orange": (240, 125, 30),
}
SHAPES = ("circle", "square", "triangle")
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six"]


@dataclass
class Obj:
    shape: str
    color: str
    box: tuple[float, float, float, float]  # pixel (x0, y0, x1, y1), right/bottom exclusive

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    def polygon(self) -> list[tuple[float, float]]:
        """Outline vertices. Circles use 12 points, which is also how they are drawn."""
        x0, y0, x1, y1 = self.box
        if self.shape == "square":
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if self.shape == "triangle":
            return [((x0 + x1) / 2, y0), (x1, y1), (x0, y1)]
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
        return [(cx + r * math.cos(2 * math.pi * k / 12), cy + r * math.sin(2 * math.pi * k / 12))
                for k in range(12)]


def _draw(draw: ImageDraw.ImageDraw, obj: Obj) -> None:
    fill = COLORS[obj.color]
    x0, y0, x1, y1 = obj.box
    if obj.shape == "circle":
        draw.ellipse([x0, y0, x1 - 1, y1 - 1], fill=fill)
    elif obj.shape == "square":
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill)
    else:
        draw.polygon([((x0 + x1) / 2, y0), (x1 - 1, y1 - 1), (x0, y1 - 1)], fill=fill)


def make_scene(rng: random.Random, n_min: int = 1, n_max: int = 5, size: int = 224,
               distinct: bool = False) -> tuple[Image.Image, list[Obj]]:
    """Place non-overlapping shapes on a light background.

    distinct=True forbids two objects with the same (colour, shape), which grounding needs so that
    "the red circle" refers to exactly one object.
    """
    bg = tuple(rng.randint(235, 255) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    n = rng.randint(n_min, n_max)
    objs: list[Obj] = []
    used: set[tuple[str, str]] = set()
    tries = 0
    while len(objs) < n and tries < 200:
        tries += 1
        s = rng.randint(size // 8, size // 4)
        x0, y0 = rng.randint(4, size - s - 4), rng.randint(4, size - s - 4)
        box = (float(x0), float(y0), float(x0 + s), float(y0 + s))
        if any(_overlap(box, o.box, margin=4) for o in objs):
            continue
        shape, color = rng.choice(SHAPES), rng.choice(list(COLORS))
        if distinct and (color, shape) in used:
            continue
        used.add((color, shape))
        obj = Obj(shape, color, box)
        _draw(draw, obj)
        objs.append(obj)
    return img, objs


def _overlap(a, b, margin: float) -> bool:
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def object_mask(obj: Obj, size: int) -> np.ndarray:
    """Exact binary mask (size, size) of one object, rendered the same way as the scene."""
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    x0, y0, x1, y1 = obj.box
    if obj.shape == "circle":
        d.ellipse([x0, y0, x1 - 1, y1 - 1], fill=255)
    elif obj.shape == "square":
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)
    else:
        d.polygon([((x0 + x1) / 2, y0), (x1 - 1, y1 - 1), (x0, y1 - 1)], fill=255)
    return np.array(m) > 0


def plural(shape: str, n: int) -> str:
    return shape if n == 1 else shape + "s"


def caption(objs: list[Obj]) -> str:
    """Deterministic caption: groups ordered by shape, then colour. "Two red circles and a blue square." """
    groups: dict[tuple[str, str], int] = {}
    for o in objs:
        groups[(o.shape, o.color)] = groups.get((o.shape, o.color), 0) + 1
    order = sorted(groups, key=lambda k: (SHAPES.index(k[0]), list(COLORS).index(k[1])))
    article = {c: "an" if c[0] in "aeiou" else "a" for c in COLORS}
    parts = [f"{article[k[1]] if groups[k] == 1 else NUMBER_WORDS[groups[k]]} {k[1]} {plural(k[0], groups[k])}"
             for k in order]
    text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return text[0].upper() + text[1:] + "."


def vqa_pair(rng: random.Random, objs: list[Obj]) -> tuple[str, str]:
    """One question about counts, colours or presence, with its exact single-word answer."""
    kind = rng.choice(["count", "count_all", "color", "exists"])
    shapes_once = [s for s in SHAPES if sum(o.shape == s for o in objs) == 1]
    if kind == "color" and shapes_once:
        s = rng.choice(shapes_once)
        return f"What color is the {s}?", next(o.color for o in objs if o.shape == s)
    if kind == "count_all":
        return "How many shapes are there?", str(len(objs))
    color, shape = _pick_pair(rng, objs)
    n = sum(o.color == color and o.shape == shape for o in objs)
    if kind == "exists":
        return f"Is there a {color} {shape}?", "yes" if n else "no"
    return f"How many {color} {shape}s are there?", str(n)


def _pick_pair(rng: random.Random, objs: list[Obj]) -> tuple[str, str]:
    """Mostly ask about something present; one time in four ask about something absent."""
    if objs and rng.random() < 0.75:
        o = rng.choice(objs)
        return o.color, o.shape
    return rng.choice(list(COLORS)), rng.choice(SHAPES)
