"""How many bins: worst-case IoU from quantisation alone, checked by brute force and by sampling (post Figure 4)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointlab.coords import bin_width, brute_force_worst_iou, quantise, worst_case_iou  # noqa: E402
from pointlab.geometry import box_iou  # noqa: E402
from pointlab.report import Report  # noqa: E402

R = Report("03_quantisation")

R("Worst case: all four edges move inward by Delta / 2, so IoU_min = ((s - Delta) / s)^2 with Delta = W / N.")
R("Brute force: every combination of the four edge errors on a 9-step grid in [-Delta/2, +Delta/2] (6,561 boxes).")
rows = []
for s, d in [(24, 3.84), (24, 1.344), (100, 10), (40, 19.2), (30, 2.0)]:
    bf, errs = brute_force_worst_iou(s, d)
    rows.append([s, d, f"{max(0, (s - d) / s) ** 2:.4f}", f"{bf:.4f}", str(errs)])
R.table(["object s (px)", "Delta (px)", "formula", "brute force", "worst edge errors (x1, y1, x2, y2)"], rows)

R("")
R("Cases from the post")
cases = [("Qwen-VL / Florence-2 grid on a 448 px image", 1000, 448, 24),
         ("Same grid on a 3840 px screenshot, 24 px icon", 1000, 3840, 24),
         ("MiniGPT-v2 0-100 grid, 3840 px screenshot, 24 px icon", 100, 3840, 24),
         ("Figure 4 default: 1000 bins, 1344 px image, 24 px object", 1000, 1344, 24),
         ("Kosmos-2 32 cells, dog filling half a 224 px photo", 32, 224, 112),
         ("Kosmos-2 32 cells, 2000 px page scan, 40 px word", 32, 2000, 40),
         ("PaliGemma 1024 bins, 2000 px page scan, 40 px word", 1024, 2000, 40)]
rows, data = [], {}
for name, n, w, s in cases:
    d = bin_width(w, n)
    iou = worst_case_iou(s, w, n)
    rows.append([name, f"{d:.2f}", f"{d / 2:.2f}", f"{iou:.2f}", "passes" if iou >= 0.5 else "can fail"])
    data[name] = {"bins": n, "image_px": w, "object_px": s, "delta": d, "worst_iou": iou}
R.table(["case", "bin width Delta (px)", "max error per edge (px)", "worst-case IoU", "Acc@0.5, worst case"], rows)

R("")
R("Average case: square objects at random positions, each edge rounded to its bin centre (20,000 boxes per row)")
rng = np.random.default_rng(0)
rows = []
for n, w in [(100, 3840), (1000, 3840), (1000, 1344), (32, 2000), (1024, 2000)]:
    for s in (24, 96):
        ious = []
        for _ in range(20000):
            x, y = rng.uniform(0, w - s, 2)
            true = (x, y, x + s, y + s)
            q = tuple(quantise(v, w, n) for v in true)
            ious.append(box_iou(true, q))
        ious = np.array(ious)
        rows.append([n, w, s, f"{ious.mean():.3f}", f"{(ious >= 0.5).mean():.3f}", f"{ious.min():.3f}",
                     f"{worst_case_iou(s, w, n):.3f}"])
        data[f"sample N={n} W={w} s={s}"] = {"mean_iou": float(ious.mean()), "acc50": float((ious >= 0.5).mean()),
                                             "min_iou": float(ious.min())}
R.table(["bins N", "image W", "object s", "mean IoU", "Acc@0.5", "min IoU seen", "worst-case bound"], rows)
R("Bin count is a resolution budget spent in pixels of the original image. Small targets on large images pay most.")
R.save(data)
