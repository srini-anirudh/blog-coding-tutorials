"""Pixel shuffle: 1024 patches -> 256 tokens with nothing lost, against 2 x 2 average pooling, which loses detail.

Writes results/03_pixel_shuffle.txt and results/03_pixel_shuffle.json.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from vencoder.common import table, write_json, write_text  # noqa: E402
from vencoder.shuffle import (  # noqa: E402
    avg_pool_tokens, patchify, pixel_shuffle, pixel_unshuffle, unpatchify, unpool_tokens,
)

STROKES = (2, 3, 5, 9, 15)


def bar_tile(size: int = 448) -> Image.Image:
    """A 448 px tile with five 3-bar gratings, one per column band, at 1 master pixel per output pixel."""
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    cell = size / len(STROKES)
    for i, w in enumerate(STROKES):
        x0 = int(i * cell + (cell - 5 * w) / 2)
        for k in range(3):
            d.rectangle((x0 + 2 * k * w, 100, x0 + 2 * k * w + w - 1, size - 100), fill=0)
    return img


def swing(a: np.ndarray, size: int = 448) -> dict[int, float]:
    """Black-to-white swing left inside each grating (1 = crisp)."""
    cell = size / len(STROKES)
    out = {}
    for i, w in enumerate(STROKES):
        x0 = int(i * cell + (cell - 5 * w) / 2)
        prof = a[150: size - 150, x0 + w // 2: x0 + 4 * w + w // 2 + 1].mean(axis=0)
        out[w] = float(prof.max() - prof.min())
    return out


def smolvlm_config() -> dict | None:
    """Read SmolVLM-256M's pixel-shuffle factor from the local Hugging Face cache, if it is there."""
    root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hits = glob.glob(os.path.join(root, "hub", "models--HuggingFaceTB--SmolVLM-256M-Instruct", "snapshots", "*", "config.json"))
    if not hits:
        return None
    c = json.load(open(hits[0]))
    return {"scale_factor": c["scale_factor"], "image_size": c["vision_config"]["image_size"],
            "patch_size": c["vision_config"]["patch_size"]}


def main() -> None:
    lines, out = [], {}

    rows = []
    configs = [("InternVL tile (InternViT-300M, d=1024)", 448, 14, 2, 1024)]
    sm = smolvlm_config()
    if sm:
        configs.append(("SmolVLM-256M crop (config.json, d=768)", sm["image_size"], sm["patch_size"], sm["scale_factor"], 768))
    configs.append(("27 x 27 grid, r = 3 (the post's SmolVLM example)", 378, 14, 3, 1152))
    for name, size, patch, r, d in configs:
        n = (size // patch) ** 2
        feats = torch.randn(1, n, d)
        y = pixel_shuffle(feats, r)
        back = pixel_unshuffle(y, r)
        rows.append([name, f"{n} x {d}", f"{y.shape[1]} x {y.shape[2]}", f"{n // y.shape[1]}x",
                     f"{(back - feats).abs().max().item():.1f}"])
    lines += ["Pixel shuffle shapes (random features), and the error after the inverse:"]
    lines += table(rows, ["input", "patches x dim", "tokens x dim", "fewer tokens", "max |inverse - input|"]) + [""]
    out["shapes"] = rows

    # What a token holds: raw 14 x 14 pixel patches of a bar tile as the 'features'.
    tile = bar_tile()
    x = patchify(tile, 14)  # 1 x 1024 x 196
    orig = unpatchify(x, 14, 32)
    shuffled = pixel_unshuffle(pixel_shuffle(x, 2), 2)
    pooled = unpool_tokens(avg_pool_tokens(x, 2), 2)
    recon = {"original": orig, "pixel shuffle, then inverse": unpatchify(shuffled, 14, 32),
             "2x2 average pool, then repeat": unpatchify(pooled, 14, 32)}
    rows, demo = [], {}
    for name, a in recon.items():
        s = swing(a)
        demo[name] = {"rmse": float(np.sqrt(((a - orig) ** 2).mean())), "swing": s}
        rows.append([name, f"{demo[name]['rmse']:.3f}", *[f"{s[w]:.2f}" for w in STROKES]])
    lines += ["448 px bar tile as raw 14 px patches (1024 tokens) compressed 4x, then expanded back:"]
    lines += table(rows, ["path", "RMSE", *[f"swing {w}px" for w in STROKES]])
    out["bar_tile"] = demo

    print("\n".join(lines))
    write_text("03_pixel_shuffle.txt", lines)
    write_json("03_pixel_shuffle.json", out)


if __name__ == "__main__":
    main()
