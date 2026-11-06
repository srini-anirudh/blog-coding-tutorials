"""Synthetic scenes where every visual fact is known exactly.

A scene is a few coloured shapes and at most one printed word on a white canvas.
Because we draw the image ourselves, we know the true colour, count, position and
text of everything in it. That makes preference labels and rewards exact.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

# Colours that scenes are drawn with. Every rejected colour in the balanced
# generator is drawn from this same list, so colour words carry no hint.
PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (214, 48, 49),
    "green": (39, 154, 72),
    "blue": (40, 96, 200),
    "yellow": (238, 190, 20),
}
# Colours that never appear in an image. Only the biased generator uses them.
OFF_PALETTE = ("purple", "orange", "pink")
SHAPES = ("circle", "square", "triangle")
WORDS = ("CAT", "DOG", "SUN", "MAP", "BOX", "CUP", "PEN", "KEY", "BUS", "HAT")

CANVAS = 512  # SmolVLM resizes to 512 anyway; drawing at 512 avoids a resample.
GRID = 3      # objects live in a 3x3 grid of cells so they never overlap.


@dataclass(frozen=True)
class Obj:
    shape: str
    color: str
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixels

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def name(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass
class Scene:
    objects: list[Obj]
    word: str | None = None
    word_box: tuple[int, int, int, int] | None = None
    size: int = CANVAS
    seed: int = 0
    meta: dict = field(default_factory=dict)

    def count(self, shape: str) -> int:
        return sum(o.shape == shape for o in self.objects)

    def with_color(self, color: str) -> list[Obj]:
        return [o for o in self.objects if o.color == color]

    def with_shape(self, shape: str) -> list[Obj]:
        return [o for o in self.objects if o.shape == shape]


def sample_scene(seed: int, min_objects: int = 2, max_objects: int = 6, with_word: bool = True) -> Scene:
    """Place objects in distinct grid cells; each (colour, shape) pair is unique."""
    rng = random.Random(seed)
    cell = CANVAS // GRID
    cells = [(r, c) for r in range(GRID) for c in range(GRID)]
    rng.shuffle(cells)
    word, word_box = None, None
    if with_word:
        r, c = cells.pop()
        word = rng.choice(WORDS)
        word_box = (c * cell + 12, r * cell + 45, (c + 1) * cell - 12, (r + 1) * cell - 45)
    n = rng.randint(min_objects, min(max_objects, len(cells)))
    identities = [(s, k) for s in SHAPES for k in PALETTE]
    rng.shuffle(identities)
    objects = []
    for (r, c), (shape, color) in zip(cells[:n], identities[:n]):
        side = rng.randint(80, 120)
        x0 = c * cell + rng.randint(8, cell - side - 8)
        y0 = r * cell + rng.randint(8, cell - side - 8)
        objects.append(Obj(shape, color, (x0, y0, x0 + side, y0 + side)))
    return Scene(objects=objects, word=word, word_box=word_box, seed=seed)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render(scene: Scene) -> Image.Image:
    """Draw the scene. Returns an RGB image of shape (size, size)."""
    img = Image.new("RGB", (scene.size, scene.size), "white")
    d = ImageDraw.Draw(img)
    for o in scene.objects:
        fill = PALETTE[o.color]
        x0, y0, x1, y1 = o.box
        if o.shape == "circle":
            d.ellipse(o.box, fill=fill)
        elif o.shape == "square":
            d.rectangle(o.box, fill=fill)
        else:
            d.polygon([((x0 + x1) / 2, y0), (x1, y1), (x0, y1)], fill=fill)
    if scene.word:
        x0, y0, x1, y1 = scene.word_box
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), scene.word, fill="black", font=_font(64), anchor="mm")
    return img


def blank_like(scene: Scene) -> Image.Image:
    """A mid-grey image of the same size: the 'no visual evidence' control."""
    return Image.new("RGB", (scene.size, scene.size), (128, 128, 128))
