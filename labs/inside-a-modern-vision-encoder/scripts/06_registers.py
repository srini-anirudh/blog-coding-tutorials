"""High-norm outlier tokens in real DINOv2 checkpoints, with and without 4 register tokens (network).

Writes results/06_registers.txt and results/06_registers.json.
    --images    images to run (taken evenly from 20 mini-ImageNet validation classes)
    --sizes     DINOv2 sizes to compare
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from vencoder.common import table, write_json, write_text  # noqa: E402
from vencoder.realdata import dinov2_input, load_dinov2, load_images, outlier_stats, token_norms  # noqa: E402

ROW_GROUPS = {"validation-00000-of-00003.parquet": list(range(20))}


def neighbour_ratio(norms: torch.Tensor, side: int) -> torch.Tensor:
    """For each image: norm of its largest token over the median norm of that token's 8 grid neighbours."""
    out = []
    for n in norms:
        g = n.reshape(side, side)
        i = int(n.argmax())
        r, c = divmod(i, side)
        nb = g[max(r - 1, 0): r + 2, max(c - 1, 0): c + 2].flatten()
        nb = nb[nb != g[r, c]]
        out.append(g[r, c] / nb.median())
    return torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=160)
    ap.add_argument("--sizes", nargs="+", default=["small", "base", "large"])
    args = ap.parse_args()

    data = load_images(ROW_GROUPS, per_group=50)
    per_class = max(1, args.images // 20)
    picked = [data[c * 50 + j] for c in range(20) for j in range(per_class)][: args.images]
    pixels = torch.stack([dinov2_input(img) for img, _ in picked])

    lines = [f"{len(picked)} mini-ImageNet validation images, 224 x 224 (256 resize, centre crop), 256 patches each.",
             "Norms are of each block's output tokens, before the final LayerNorm.",
             "outlier = patch token with norm > 3x its image's median patch norm.",
             "peak block = the block where the no-register model's largest token stands out most; the",
             "registers model is read at that same block. 'last' = the final block.", ""]
    rows, out = [], {}
    for size in args.sizes:
        peak = None
        for reg in (False, True):
            name = f"facebook/dinov2-{'with-registers-' if reg else ''}{size}"
            t0 = time.time()
            model, n_reg = load_dinov2(name)
            norms, reg_norms = token_norms(model, pixels, n_reg)  # layers x images x patches
            dt = time.time() - t0
            per_layer = [outlier_stats(n, 3.0) for n in norms]
            if peak is None:
                peak = max(range(len(per_layer)), key=lambda i: per_layer[i]["max_over_median_mean"])
            at, last = per_layer[peak], per_layer[-1]
            nb = neighbour_ratio(norms[peak], 16)
            reg_norm = float(reg_norms[peak].mean()) if n_reg else None
            out[name] = {"registers": n_reg, "blocks": len(per_layer), "peak_block": peak + 1, "at_peak": at,
                         "last_block": last, "neighbour_ratio_median_at_peak": float(nb.median()),
                         "median_patch_norm_at_peak": float(norms[peak].median()),
                         "register_norm_mean_at_peak": reg_norm, "seconds": dt,
                         "outlier_fraction_by_block": [round(x["outlier_fraction"], 5) for x in per_layer]}
            rows.append([name.split("/")[1], n_reg, f"{peak + 1} of {len(per_layer)}",
                         f"{float(norms[peak].median()):.1f}", f"{at['outlier_fraction']:.2%}",
                         f"{at['images_with_outliers']:.0%}", f"{at['max_over_median_mean']:.1f}",
                         f"{float(nb.median()):.1f}", f"{last['outlier_fraction']:.2%}",
                         "-" if reg_norm is None else f"{reg_norm:.1f}"])
            print(rows[-1], f"{dt:.0f}s", flush=True)
            del model
    lines += table(rows, ["model", "registers", "peak block", "median patch norm", "> 3x", "images with > 3x",
                          "max / median", "max / its 8 neighbours", "> 3x, last block", "mean register norm"])

    print("\n".join(lines))
    write_text("06_registers.txt", lines)
    write_json("06_registers.json", {"images": len(picked), "models": out})


if __name__ == "__main__":
    main()
