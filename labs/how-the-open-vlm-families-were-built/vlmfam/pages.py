"""DeepSeek-OCR's question: how many text tokens does one vision token carry?

We render real text (WikiText-2) onto a square page at several font sizes, count the text tokens that fit with
DeepSeek-OCR's own tokenizer, and divide by the vision tokens each DeepEncoder mode spends on the page.

DeepEncoder's token count follows from its design: 16 px patches, then a 16x convolutional compressor, so a
side x side input costs (side / 16)^2 / 16 tokens. The four single-view modes are 512, 640, 1024 and 1280 px.
We do not run the model, so decoding precision is not measured here; the post quotes it from the paper.
"""

from __future__ import annotations

import re

from PIL import Image, ImageDraw, ImageFont

MODES = {"Tiny": 512, "Small": 640, "Base": 1024, "Large": 1280}


def deepencoder_tokens(side: int, patch: int = 16, compress: int = 16) -> int:
    return (side // patch) ** 2 // compress


def clean_wikitext(lines: list[str]) -> str:
    """Join WikiText-2 raw lines into plain prose: drop headings and undo its spacing conventions."""
    keep = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("=")]
    text = " ".join(keep)
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r" ([.,;:!?)'])", r"\1", text).replace("( ", "(")
    return re.sub(r"\s+", " ", text)


def font(size: int):
    return ImageFont.load_default(size=size)


def fill_page(text: str, font_size: int, side: int = 1024, margin: int = 32, line_gap: float = 1.25):
    """Word-wrap `text` onto a side x side page. Returns (the text that fit, the rendered page)."""
    f = font(font_size)
    page = Image.new("L", (side, side), 255)
    draw = ImageDraw.Draw(page)
    width = side - 2 * margin
    step = int(font_size * line_gap)
    y, words, i, fitted = margin, text.split(" "), 0, []
    while y + step <= side - margin and i < len(words):
        line = words[i]
        i += 1
        while i < len(words) and draw.textlength(line + " " + words[i], font=f) <= width:
            line += " " + words[i]
            i += 1
        draw.text((margin, y), line, fill=0, font=f)
        fitted.append(line)
        y += step
    return " ".join(fitted), page
