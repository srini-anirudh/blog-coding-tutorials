"""Rendered text for OCR and document questions. The string drawn is the label, so it is exact."""

from __future__ import annotations

import random
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

WORDS = (
    "apple river stone cloud green light paper music water house table chair window garden "
    "bread honey ocean tiger horse lemon pearl silver amber cedar maple north south quiet "
    "rapid storm field night morning winter summer spring autumn bridge castle forest island "
    "market planet rocket signal travel valley yellow orange purple marble copper engine "
    "harbor meadow pencil rabbit shadow simple target ticket violet wonder anchor basket "
    "candle dragon falcon guitar helmet jacket kitten ladder magnet napkin oyster pepper "
    "radio salad tomato velvet walnut zebra camera desert friend circle square number letter"
).split()

FIRST = "Ana Ben Chloe Dev Elena Farid Grace Hiro Ines Jon Kavya Liam Mira Noor Omar Priya".split()
LAST = "Lee Shah Brown Rossi Kim Silva Novak Patel Okafor Garcia Chen Weber Rao Haddad".split()


@lru_cache(maxsize=None)
def font(size: int) -> ImageFont.FreeTypeFont:
    """Pillow's bundled scalable font, so rendering is identical on every machine."""
    return ImageFont.load_default(size=size)


def make_ocr(rng: random.Random) -> tuple[Image.Image, str]:
    """1-3 words on a wide light canvas. Wide images are where tiling earns its tokens."""
    text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 3)))
    w, h = rng.choice([(448, 224), (448, 160), (336, 224)])
    img = Image.new("RGB", (w, h), tuple(rng.randint(225, 255) for _ in range(3)))
    draw = ImageDraw.Draw(img)
    size = rng.randint(30, 46)
    f = font(size)
    while draw.textlength(text, font=f) > w - 16 and size > 14:  # shrink until the line fits
        size -= 2
        f = font(size)
    tw = draw.textlength(text, font=f)
    x = rng.randint(8, max(8, int(w - tw - 8)))
    y = rng.randint(8, h - size - 12)
    draw.text((x, y), text, fill=tuple(rng.randint(0, 70) for _ in range(3)), font=f)
    return img, text


def document_fields(rng: random.Random) -> dict[str, str]:
    return {
        "Invoice No": str(rng.randint(1000, 9999)),
        "Date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "Name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        "Items": str(rng.randint(1, 9)),
        "Total": f"{rng.randint(5, 99)}.{rng.choice(['00', '25', '50', '75', '90'])}",
    }


DOC_QUESTIONS = {
    "Invoice No": "What is the invoice number?",
    "Date": "What is the date on the invoice?",
    "Name": "Who is the invoice for?",
    "Items": "How many items are on the invoice?",
    "Total": "What is the total?",
}


def make_document(rng: random.Random, size: int = 448) -> tuple[Image.Image, dict[str, str]]:
    """A one-page invoice: a title and five "Key: value" lines in small type.

    At 448 px the body text is about 20 px tall. Squashed to one 224 px tile it becomes about
    10 px, which is the regime where resolution decides whether the answer is readable at all.
    """
    fields = document_fields(rng)
    img = Image.new("RGB", (size, size), (250, 250, 247))
    draw = ImageDraw.Draw(img)
    x = rng.randint(24, 60)
    y = rng.randint(20, 50)
    draw.text((x, y), "INVOICE", fill=(20, 20, 20), font=font(34))
    y += 60
    draw.line([(x, y - 10), (size - 30, y - 10)], fill=(150, 150, 150), width=2)
    body = font(rng.randint(20, 24))
    for key, value in fields.items():
        draw.text((x, y), f"{key}: {value}", fill=(30, 30, 30), font=body)
        y += rng.randint(42, 56)
    return img, fields
