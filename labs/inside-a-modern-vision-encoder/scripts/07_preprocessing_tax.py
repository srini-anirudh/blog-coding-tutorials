"""The preprocessing tax on a real encoder: DINOv2 kNN accuracy with one preprocessing step broken at a time (network).

Writes results/07_preprocessing_tax.txt and results/07_preprocessing_tax.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vencoder.common import table, write_json, write_text  # noqa: E402
from vencoder.realdata import dinov2_input, encode, knn_accuracy, load_dinov2, load_images  # noqa: E402

ROW_GROUPS = {"validation-00000-of-00003.parquet": list(range(20))}
CONDITIONS = {
    "correct: bicubic AA, crop, ImageNet constants": {},
    "SigLIP constants (0.5 / 0.5)": {"norm": "SigLIP (0.5)"},
    "OpenAI CLIP constants": {"norm": "OpenAI CLIP"},
    "torch bicubic, antialias=False": {"resize": "torch_no_aa"},
    "PIL nearest": {"resize": "pil_nearest"},
    "squash to 224 instead of crop": {"crop": False},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["facebook/dinov2-small", "facebook/dinov2-base"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    data = load_images(ROW_GROUPS, per_group=50)
    labels = torch.tensor([lab for _, lab in data])
    lines = [f"{len(data)} mini-ImageNet validation images, {len(set(labels.tolist()))} classes, "
             f"leave-one-out {args.k}-NN top-1 on the CLS feature. One preprocessing step changed per row.", ""]
    out, rows = {}, []
    for name in args.models:
        model, _ = load_dinov2(name)
        ref = None
        for cond, kw in CONDITIONS.items():
            t0 = time.time()
            pixels = torch.stack([dinov2_input(img, **kw) for img, _ in data])
            cls = encode(model, pixels)["cls"]
            if ref is None:
                ref = cls
            acc = knn_accuracy(cls, labels, args.k)
            cos = float(F.cosine_similarity(cls, ref, dim=-1).mean())
            out.setdefault(name, {})[cond] = {"knn_top1": acc, "cos_to_correct": cos, "seconds": time.time() - t0}
            rows.append([name.split("/")[1], cond, f"{acc:.3f}", f"{cos:.3f}"])
            print(rows[-1], flush=True)
        del model
    lines += table(rows, ["model", "preprocessing", "kNN top-1", "mean cos to correct CLS"])

    print("\n".join(lines))
    write_text("07_preprocessing_tax.txt", lines)
    write_json("07_preprocessing_tax.json", {"images": len(data), "k": args.k, "results": out})


if __name__ == "__main__":
    main()
