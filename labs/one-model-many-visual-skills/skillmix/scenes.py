"""Synthetic scenes: a 3 x 3 grid of 8 x 8 cells, each empty or holding one coloured object.

The world has one built-in co-occurrence: a fork usually sits next to a plate. That bias is what a
biased "teacher" captioner later copies into its captions (script 05).

Forks are always grey. So "what color is the fork ?" can be answered without looking, which gives
script 06 a question that does not need the image.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

GRID = 3
CELL = 8
IMAGE_SIZE = GRID * CELL  # 24 x 24 pixels
OBJECTS = ["plate", "fork", "cup", "ball", "book"]
COLOURS = {"red": (1.0, 0.15, 0.15), "green": (0.15, 0.85, 0.2), "blue": (0.2, 0.3, 1.0),
           "yellow": (1.0, 0.9, 0.1), "grey": (0.6, 0.6, 0.6)}
PAINTABLE = ["red", "green", "blue", "yellow"]  # every object except the fork takes one of these
POSITIONS = ["top left", "top", "top right", "left", "middle", "right", "bottom left", "bottom", "bottom right"]
P_FORK_GIVEN_PLATE = 0.7


@dataclass(frozen=True)
class Item:
    obj: str
    colour: str
    cell: int  # 0..8 in reading order

    @property
    def where(self) -> str:
        return POSITIONS[self.cell]


@dataclass(frozen=True)
class Scene:
    items: tuple[Item, ...]  # sorted by cell, at most one of each object type

    def has(self, obj: str) -> bool:
        return any(it.obj == obj for it in self.items)

    def get(self, obj: str) -> Item:
        return next(it for it in self.items if it.obj == obj)

    def count(self, colour: str | None = None) -> int:
        return sum(1 for it in self.items if colour is None or it.colour == colour)


def _mask(obj: str) -> torch.Tensor:
    """(8, 8) binary stamp of one object shape."""
    m = torch.zeros(CELL, CELL)
    yy, xx = torch.meshgrid(torch.arange(CELL) - 3.5, torch.arange(CELL) - 3.5, indexing="ij")
    r = (yy ** 2 + xx ** 2).sqrt()
    if obj == "plate":  # ring
        m[(r > 2.0) & (r < 3.6)] = 1
    elif obj == "fork":  # three prongs on a handle
        m[1:4, [2, 4, 6]] = 1
        m[3, 2:7] = 1
        m[4:7, 4] = 1
    elif obj == "cup":  # filled square with a handle
        m[2:6, 1:5] = 1
        m[3:5, 6] = 1
        m[3, 5] = m[4, 5] = 1
    elif obj == "ball":  # small filled disk
        m[r < 2.2] = 1
    elif obj == "book":  # wide rectangle with a spine
        m[2:6, 0:8] = 1
        m[2:6, 1] = 0
    return m


_STAMPS = {o: _mask(o) for o in OBJECTS}


def render(scene: Scene) -> torch.Tensor:
    """(3, 24, 24) float image on a dark background."""
    img = torch.full((3, IMAGE_SIZE, IMAGE_SIZE), 0.05)
    for it in scene.items:
        r, c = divmod(it.cell, GRID)
        rgb = torch.tensor(COLOURS[it.colour]).view(3, 1, 1)
        stamp = _STAMPS[it.obj]
        patch = img[:, r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
        img[:, r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = torch.where(stamp > 0, rgb, patch)
    return img


def sample_scene(rng: random.Random, min_items: int = 1, max_items: int = 4,
                 require: str | None = None, forbid: str | None = None) -> Scene:
    """A random scene. A fork joins a plate with probability P_FORK_GIVEN_PLATE."""
    while True:
        n = rng.randint(min_items, max_items)
        objs = rng.sample([o for o in OBJECTS if o != "fork"], k=min(n, 4))
        if "plate" in objs and rng.random() < P_FORK_GIVEN_PLATE:
            objs.append("fork")
        elif "plate" not in objs and rng.random() < 0.1:
            objs.append("fork")
        objs = objs[:max_items]
        if require and require not in objs:
            continue
        if forbid and forbid in objs:
            continue
        cells = rng.sample(range(GRID * GRID), k=len(objs))
        items = [Item(o, "grey" if o == "fork" else rng.choice(PAINTABLE), c) for o, c in zip(objs, cells)]
        return Scene(tuple(sorted(items, key=lambda it: it.cell)))


def recolour(scene: Scene, old: str, new: str) -> Scene:
    """Counterfactual edit: repaint the first object of colour `old` with colour `new`."""
    items = list(scene.items)
    for i, it in enumerate(items):
        if it.colour == old:
            items[i] = Item(it.obj, new, it.cell)
            break
    return Scene(tuple(items))
