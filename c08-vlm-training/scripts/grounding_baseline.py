#!/usr/bin/env python
"""No-model control for grounding: one constant box and one constant point for every question.

Run from the repo root:  python c08-vlm-training/scripts/grounding_baseline.py c08-vlm-training

The constant box is chosen on training samples 0..599 (the box on the 32-bin grid with the best mean IoU, and the
one with the most hits at IoU 0.5). Both are then scored on the same 128 test samples the models were scored on
(70 boxes, 29 points, 29 outlines). The upper bound picks the constant on the test samples themselves, which
no real baseline could do. Outlines are not scored here.
"""
import sys
from pathlib import Path

import numpy as np

folder = Path(sys.argv[1])
sys.path.insert(0, str(folder))
from vlmtrain.data import make_example  # noqa: E402
from vlmtrain.device import describe_hardware, cap_threads, pick_device  # noqa: E402

cap_threads()
BINS, SIZE = 32, 224.0
N_TRAIN, N_TEST = 600, 128


def collect(split, n):
    box, pt = [], []
    for i in range(n):
        m = make_example("grounding", i, split)["meta"]
        (box if m["kind"] == "box" else pt if m["kind"] == "point" else []).append(m["box"])
    return np.array(box, float), np.array(pt, float)


def grid_boxes():
    """Every box on the bin grid, decoded the way decode_box decodes: bin a to bin b spans a's left edge to b's right edge."""
    a = np.arange(BINS)
    x0, x1 = np.meshgrid(a, a, indexing="ij")
    ok = x0 <= x1
    x0, x1 = x0[ok], x1[ok]
    s = SIZE / BINS
    x0, x1 = x0 * s, (x1 + 1) * s
    X0, Y0 = np.meshgrid(x0, x0, indexing="ij")  # reuse the same pairs for y
    X1, Y1 = np.meshgrid(x1, x1, indexing="ij")
    return np.stack([X0.ravel(), Y0.ravel(), X1.ravel(), Y1.ravel()], 1)


def iou_matrix(pred, true, chunk=20000):
    out = np.empty((len(pred), len(true)))
    for s in range(0, len(pred), chunk):
        p = pred[s:s + chunk, None, :]
        ix = np.clip(np.minimum(p[..., 2], true[:, 2]) - np.maximum(p[..., 0], true[:, 0]), 0, None)
        iy = np.clip(np.minimum(p[..., 3], true[:, 3]) - np.maximum(p[..., 1], true[:, 1]), 0, None)
        inter = ix * iy
        area = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1]) + (true[:, 2] - true[:, 0]) * (true[:, 3] - true[:, 1]) - inter
        out[s:s + chunk] = inter / area
    return out


def point_hits(pts, boxes):
    return ((pts[:, None, 0] >= boxes[:, 0]) & (pts[:, None, 0] <= boxes[:, 2]) &
            (pts[:, None, 1] >= boxes[:, 1]) & (pts[:, None, 1] <= boxes[:, 3])).astype(float)


print(describe_hardware(pick_device()))
tr_box, tr_pt = collect("train", N_TRAIN)
te_box, te_pt = collect("test", N_TEST)
print(f"train samples 0..{N_TRAIN - 1}: {len(tr_box)} boxes, {len(tr_pt)} points; test samples 0..{N_TEST - 1}: {len(te_box)} boxes, {len(te_pt)} points")

cand = grid_boxes()
I_tr, I_te = iou_matrix(cand, tr_box), iou_matrix(cand, te_box)
fmt = lambda b: "(" + ", ".join(f"{v:.0f}" for v in b) + ")"
print(f"\n{'constant box':<46}{'mean IoU':>9}{'hits at 0.5':>13}")
for name, pick in [("tuned on train, best mean IoU", I_tr.mean(1).argmax()),
                   ("tuned on train, most hits at IoU 0.5", (I_tr >= 0.5).sum(1).argmax()),
                   ("tuned on test, best mean IoU (upper bound)", I_te.mean(1).argmax()),
                   ("tuned on test, most hits at IoU 0.5 (upper bound)", (I_te >= 0.5).sum(1).argmax())]:
    print(f"{name:<46}{I_te[pick].mean():>9.3f}{int((I_te[pick] >= 0.5).sum()):>9} of {len(te_box)}   box {fmt(cand[pick])}")

s = SIZE / BINS
centres = np.array([((i + 0.5) * s, (j + 0.5) * s) for i in range(BINS) for j in range(BINS)])
H_tr, H_te = point_hits(centres, tr_pt), point_hits(centres, te_pt)
print(f"\n{'constant point':<46}{'hits':>13}")
for name, pick in [("tuned on train, most hits", H_tr.sum(1).argmax()), ("tuned on test, most hits (upper bound)", H_te.sum(1).argmax())]:
    print(f"{name:<46}{int(H_te[pick].sum()):>9} of {len(te_pt)}   point {fmt(centres[pick])}")
