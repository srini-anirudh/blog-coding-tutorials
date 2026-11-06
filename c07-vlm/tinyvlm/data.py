"""A synthetic caption set: coloured shapes on a plain canvas, with exact captions.

Synthetic data makes the label exact, so we can score a generated caption by parsing it:
did the model get the count, the colour and the shape right?

Each example uses the unified multimodal format that C8 also uses:
    {"images": [PIL.Image], "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", ...}]},
                                         {"role": "assistant", "content": "Two red circles."}]}
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

COLORS = {
    "red": (220, 40, 40),
    "green": (40, 170, 70),
    "blue": (40, 80, 220),
    "yellow": (235, 200, 30),
    "purple": (140, 60, 180),
    "orange": (245, 140, 30),
}
SHAPES = ("circle", "square", "triangle")
COUNT_WORDS = {1: "One", 2: "Two", 3: "Three"}
PROMPTS = ("Describe the image.", "What do you see?", "Write a short caption.")


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: int, cy: int, r: int, fill: tuple[int, int, int]) -> None:
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)
    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fill)
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)


def render(count: int, color: str, shape: str, rng: random.Random, size: int = 224) -> Image.Image:
    """Draw `count` non-overlapping shapes of one colour on an off-white canvas."""
    shade = rng.randint(235, 255)
    image = Image.new("RGB", (size, size), (shade, shade, shade))
    draw = ImageDraw.Draw(image)
    placed: list[tuple[int, int, int]] = []
    while len(placed) < count:
        r = rng.randint(18, 36)
        cx, cy = rng.randint(r + 4, size - r - 4), rng.randint(r + 4, size - r - 4)
        if all((cx - x) ** 2 + (cy - y) ** 2 > (r + rr + 6) ** 2 for x, y, rr in placed):
            placed.append((cx, cy, r))
            draw_shape(draw, shape, cx, cy, r, COLORS[color])
    return image


def caption(count: int, color: str, shape: str) -> str:
    return f"{COUNT_WORDS[count]} {color} {shape}{'s' if count > 1 else ''}."


def make_example(index: int, seed: int = 0) -> dict:
    """Deterministic: the same (index, seed) always gives the same picture and caption."""
    rng = random.Random(seed * 1_000_003 + index)
    count, color, shape = rng.randint(1, 3), rng.choice(list(COLORS)), rng.choice(SHAPES)
    return {
        "images": [render(count, color, shape, rng)],
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": rng.choice(PROMPTS)}]},
            {"role": "assistant", "content": caption(count, color, shape)},
        ],
        "target": {"count": count, "color": color, "shape": shape},
    }


def parse_caption(text: str) -> dict:
    """Read count, colour and shape back out of free text (None where nothing matches)."""
    words = text.lower().replace(".", " ").replace(",", " ").split()
    counts = {w.lower(): n for n, w in COUNT_WORDS.items()} | {"a": 1, "an": 1, "1": 1, "2": 2, "3": 3}
    found = {"count": None, "color": None, "shape": None}
    for w in words:
        if found["count"] is None and w in counts:
            found["count"] = counts[w]
        if found["color"] is None and w in COLORS:
            found["color"] = w
        if found["shape"] is None and w.rstrip("s") in SHAPES:
            found["shape"] = w.rstrip("s")
    return found


def score(predictions: list[str], targets: list[dict]) -> dict[str, float]:
    """Fraction correct per attribute, and for all three at once."""
    parsed = [parse_caption(p) for p in predictions]
    result = {k: sum(p[k] == t[k] for p, t in zip(parsed, targets)) / len(targets) for k in ("count", "color", "shape")}
    result["all"] = sum(p == t for p, t in zip(parsed, targets)) / len(targets)
    return result
