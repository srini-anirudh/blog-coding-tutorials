"""What each resolution strategy costs: tokens per image, and vision encoder throughput.

    python c08-vlm-training/benchmarks/bench_resolution.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vlmtrain.c07_adapter import build_c07  # noqa: E402
from vlmtrain.device import laptop_threads, pick_device, sync  # noqa: E402
from vlmtrain.resolution import n_tiles, preprocess  # noqa: E402

SIZES = {"scene 224x224": (224, 224), "ocr 336x224": (336, 224), "ocr 448x160": (448, 160), "ocr 448x224": (448, 224),
         "document 448x448": (448, 448)}
TOKENS_PER_TILE = 49


def main() -> None:
    laptop_threads(2)
    print(f"{'image':<20}{'resize':>8}{'pad':>8}{'tile':>8}   (visual tokens)")
    tbl = {}
    for name, (w, h) in SIZES.items():
        row = {s: n_tiles(w, h, s) * TOKENS_PER_TILE for s in ("resize", "pad", "tile")}
        tbl[name] = row
        print(f"{name:<20}{row['resize']:>8}{row['pad']:>8}{row['tile']:>8}")

    device = pick_device()
    _, model, _ = build_c07()
    model.to(device).eval()
    doc = Image.new("RGB", (448, 448), "white")
    print(f"\nEncoding one 448x448 document on {device}")
    print(f"{'strategy':<10}{'tiles':>7}{'ms/image':>10}")
    timed = {}
    for s in ("resize", "tile"):
        x = preprocess(doc, s).to(device)
        with torch.no_grad():
            model.get_image_features(x)
            sync(device)
            t0 = time.time()
            for _ in range(5):
                model.get_image_features(x)
            sync(device)
        ms = (time.time() - t0) / 5 * 1000
        timed[s] = {"tiles": x.shape[0], "ms": ms}
        print(f"{s:<10}{x.shape[0]:>7}{ms:>10.0f}")
    (ROOT / "results" / "bench_resolution.json").write_text(json.dumps({"tokens": tbl, "timed": timed, "device": str(device)}, indent=1))


if __name__ == "__main__":
    main()
