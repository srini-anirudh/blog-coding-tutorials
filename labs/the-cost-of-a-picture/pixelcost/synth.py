"""Synthetic test pictures with exact answers: a large coloured shape (a scene question) and a small printed
number (a reading question). The number sits in the top or the bottom band, so position bias can show."""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

COLORS = {"red": (220, 30, 30), "green": (30, 160, 60), "blue": (30, 70, 220), "yellow": (235, 200, 20),
          "purple": (140, 40, 170)}


@dataclass
class Sample:
    image: Image.Image
    color: str
    number: str
    band: str          # "top" or "bottom": where the number is printed


def make_sample(seed: int, side: int = 672, band: str | None = None, font_px: int = 30) -> Sample:
    rng = random.Random(seed)
    band = band or rng.choice(["top", "bottom"])
    color = rng.choice(sorted(COLORS))
    number = str(rng.randint(1000, 9999))
    img = Image.new("RGB", (side, side), (245, 245, 240))
    d = ImageDraw.Draw(img)
    s = rng.randint(side // 4, side // 3)                          # shape in the middle band
    x0 = rng.randint(side // 10, side - side // 10 - s)
    y0 = rng.randint(int(side * 0.30), int(side * 0.70) - s)
    box = (x0, y0, x0 + s, y0 + s)
    (d.ellipse if rng.random() < 0.5 else d.rectangle)(box, fill=COLORS[color])
    font = ImageFont.load_default(size=font_px)
    tw = d.textlength(number, font=font)
    tx = rng.randint(10, int(side - tw - 10))
    ty = rng.randint(10, int(side * 0.18)) if band == "top" else rng.randint(int(side * 0.80), side - font_px - 12)
    d.text((tx, ty), number, fill=(15, 15, 15), font=font)
    return Sample(img, color, number, band)


def solid_shape(color: str, side: int = 224, circle: bool = True) -> Image.Image:
    """A single shape on a plain background, for the prefix-cache demo."""
    img = Image.new("RGB", (side, side), (245, 245, 240))
    d = ImageDraw.Draw(img)
    m = side // 5
    (d.ellipse if circle else d.rectangle)((m, m, side - m, side - m), fill=COLORS[color])
    return img


QUESTIONS = {"color": "What color is the large shape? Answer with one word.",
             "number": "What is the number printed in the image? Answer with the number only."}


def is_correct(kind: str, sample: Sample, answer: str) -> bool:
    a = answer.strip().lower()
    return sample.color in a if kind == "color" else sample.number in a.replace(",", "").replace(" ", "")
