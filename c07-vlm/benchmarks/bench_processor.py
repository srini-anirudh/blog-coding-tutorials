"""Where does preprocessing time go? Ours against Hugging Face, split into resize and normalise (CPU)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers.models.siglip.image_processing_pil_siglip import SiglipImageProcessorPil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyvlm.image_processing import TinyVLMImageProcessor, normalize, resize  # noqa: E402

torch.set_num_threads(2)


def timed(fn, repeats: int = 20) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1000


rng = np.random.default_rng(0)
print(f"{'picture':>12} {'resize ms':>10} {'normalise ms':>13} {'ours ms':>9} {'hf ms':>9}")
hf = SiglipImageProcessorPil(size={"height": 224, "width": 224}, resample=3)
ours = TinyVLMImageProcessor()
for w, h in [(224, 224), (640, 480), (1920, 1080)]:
    image = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    small = resize(image, 224)
    print(
        f"{w:>6} x {h:<4} {timed(lambda: resize(image, 224)):>10.2f} {timed(lambda: normalize(small)):>13.2f} "
        f"{timed(lambda: ours([image])):>9.2f} {timed(lambda: hf(image, return_tensors='pt')):>9.2f}"
    )
tiler = TinyVLMImageProcessor(strategy="tile", max_tiles=6)
image = Image.fromarray(rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8))
print(f"tile strategy, 1920 x 1080, {tiler.num_crops_for_size(1920, 1080)} crops: {timed(lambda: tiler([image]), 5):.2f} ms")
