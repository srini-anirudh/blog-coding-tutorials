"""Matplotlib bar charts with known values, for chart question answering."""

from __future__ import annotations

import io
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

CATEGORIES = ["cat", "dog", "fox", "owl", "bee", "elk", "yak", "emu"]
BAR_COLORS = ["#3a6fd8", "#d9534f", "#2e9e5b", "#e08a1e", "#7a4fb5"]


def make_chart(rng: random.Random, size: int = 224) -> tuple[Image.Image, dict[str, int]]:
    """3-5 bars with integer heights 1-9 on a gridded axis, so every value is readable."""
    n = rng.randint(3, 5)
    labels = rng.sample(CATEGORIES, n)
    values = {lab: rng.randint(1, 9) for lab in labels}
    dpi = 56
    fig, ax = plt.subplots(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax.bar(labels, [values[k] for k in labels], color=rng.choice(BAR_COLORS))
    ax.set_ylim(0, 10)
    ax.set_yticks(range(0, 11, 1))
    ax.grid(axis="y", color="#cccccc", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((size, size)), values


def chart_question(rng: random.Random, values: dict[str, int]) -> tuple[str, str]:
    """Either read one bar or name the tallest one (only asked when the maximum is unique)."""
    top = max(values.values())
    leaders = [k for k, v in values.items() if v == top]
    if len(leaders) == 1 and rng.random() < 0.3:
        return "Which category has the highest value?", leaders[0]
    lab = rng.choice(list(values))
    return f"What is the value of {lab}?", str(values[lab])
