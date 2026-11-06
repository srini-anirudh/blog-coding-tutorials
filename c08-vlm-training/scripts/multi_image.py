"""Several images in one example: tiles, <image> slots and what they do to a token budget.

    python scripts/multi_image.py

No training task in this folder uses more than one image. This script shows that the processor, the
packer and C7's model handle it, and what it costs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmtrain.c07_adapter import build_c07  # noqa: E402
from vlmtrain.data import make_example, sample  # noqa: E402
from vlmtrain.device import cap_threads, describe_hardware, pick_device  # noqa: E402
from vlmtrain.packing import first_fit_decreasing  # noqa: E402


def main() -> None:
    cap_threads()
    device = pick_device()
    _, model, proc = build_c07()
    model.to(device).eval()
    print(describe_hardware(device))
    scene = make_example("caption", 0)["images"][0]
    ocr = make_example("ocr", 0)["images"][0]
    doc = make_example("doc", 0)["images"][0]
    examples = {
        "one scene": [scene],
        "scene + ocr": [scene, ocr],
        "scene + doc": [scene, doc],
        "doc + doc + doc": [doc, doc, doc],
    }
    print(f"{'images':<18}{'sizes':<28}{'tiles_per_image':<17}{'slots':>6}{'pixel_values':>20}{'logits':>20}")
    lengths = {}
    for name, images in examples.items():
        s = sample("vqa", images, "Compare the images.", "They differ.")
        e = proc.encode(s)
        b = proc.collate([e])
        with torch.no_grad():
            logits = model(b["input_ids"].to(device), b["pixel_values"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
        sizes = ", ".join("x".join(map(str, im.size)) for im in images)
        slots = e.input_ids.count(proc.image_token_id)
        print(f"{name:<18}{sizes:<28}{str(e.tiles_per_image):<17}{slots:>6}"
              f"{str(tuple(b['pixel_values'].shape)):>20}{str(tuple(logits.shape[:2])):>20}")
        lengths[name] = len(e.input_ids)

    print("\nRows needed for 32 examples of each kind, by token budget:")
    print(f"{'images':<18}{'tokens':>7}{'budget 512':>12}{'budget 1024':>13}")
    for name, n in lengths.items():
        row = []
        for budget in (512, 1024):
            row.append("too long" if n > budget else str(len(first_fit_decreasing([n] * 32, budget))))
        print(f"{name:<18}{n:>7}{row[0]:>12}{row[1]:>13}")


if __name__ == "__main__":
    main()
