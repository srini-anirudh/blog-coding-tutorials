"""Measuring grounding: inside tests, gIoU vs cIoU, boundary IoU, the GIoU name clash, many regions or none,
granularity mismatch and temporal IoU (post section "Measuring grounding")."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointlab import geometry as G  # noqa: E402
from pointlab.report import Report  # noqa: E402
from pointlab.scoring import boundary_iou, generalised_iou, giou_ciou, match_f1, sequence_f1  # noqa: E402

R = Report("07_metrics")
data = {}

# ---- points: the inside test and the pointing game ---------------------------------------------
W, H = 600, 330
band = G.polygon_mask(G.figure2_band(256), W, H)
cen = G.centroid(band)
R("Points. The Figure 2 band, scored with an inside test and with the pointing game's 15 px tolerance")
for name, pt in [("centroid", cen), ("point on the band", (300, 102))]:
    R(f"  {name:18s} ({pt[0]:.1f}, {pt[1]:.1f})  inside: {G.point_in_mask(pt, band)}   "
      f"within 15 px: {G.point_in_mask(pt, band, tolerance=15)}")
ys, xs = np.nonzero(band)
d = np.hypot(xs + 0.5 - cen[0], ys + 0.5 - cen[1]).min()
R(f"  the centroid is {d:.1f} px from the nearest band pixel, so even the tolerance does not save it")
data["centroid_distance_px"] = float(d)

# ---- gIoU vs cIoU -------------------------------------------------------------------------------
R("")
R("Masks. gIoU averages per-example IoU; cIoU pools pixels, so large objects dominate it")
S = 128


def square(x, y, s):
    m = np.zeros((S, S), bool)
    m[y:y + s, x:x + s] = True
    return m


gts, good_large = [], []
for i in range(2):                        # two large objects, predicted almost perfectly
    gts.append(square(10, 10, 100))
    good_large.append(square(12, 12, 100))
for i in range(8):                        # eight small objects, predicted badly (shifted by most of their size)
    gts.append(square(40, 40, 8))
    good_large.append(square(46, 46, 8))
g, c = giou_ciou(good_large, gts)
R(f"  2 large objects right, 8 small objects wrong:  gIoU {g:.3f}   cIoU {c:.3f}")
rev = [square(40, 40, 60) for _ in range(2)] + [square(40, 40, 8) for _ in range(8)]
g2, c2 = giou_ciou(rev, gts)
R(f"  2 large objects wrong, 8 small objects right:  gIoU {g2:.3f}   cIoU {c2:.3f}")
data["giou_ciou"] = {"large_right": [g, c], "small_right": [g2, c2]}

# ---- boundary IoU ------------------------------------------------------------------------------
R("")
R("Boundary IoU (a 2 px band inside each contour) exposes edge errors that whole-mask IoU hides")
yy, xx = np.mgrid[0:S, 0:S] + 0.5
ang = np.arctan2(yy - 64, xx - 64)
rad = np.hypot(xx - 64, yy - 64)
gt = rad <= 40
ragged = rad <= 40 + 3 * np.sign(np.sin(24 * ang))
shifted = np.hypot(xx - 67, yy - 64) <= 40
for name, p in [("ragged edge (radius 40 +- 3 px)", ragged), ("smooth, shifted by 3 px", shifted)]:
    R(f"  {name:32s} mask IoU {G.mask_iou(p, gt):.3f}   boundary IoU {boundary_iou(p, gt):.3f}")
    data[f"boundary {name}"] = [G.mask_iou(p, gt), boundary_iou(p, gt)]

# ---- the gIoU / GIoU name clash ----------------------------------------------------------------
R("")
R("'gIoU' in segmentation is a mean of IoUs. 'GIoU' in detection is generalised IoU (Rezatofighi et al.)")
for a, b in [((0, 0, 10, 10), (5, 0, 15, 10)), ((0, 0, 10, 10), (20, 0, 30, 10)), ((0, 0, 10, 10), (40, 0, 50, 10))]:
    R(f"  boxes {a} and {b}:  IoU {G.box_iou(a, b):.3f}   GIoU {generalised_iou(a, b):+.3f}")
R("  GIoU goes negative for disjoint boxes and keeps falling with distance; IoU stays at 0.")

# ---- many regions, or none ---------------------------------------------------------------------
R("")
R("Many regions: 'which months were over budget?' points at three table cells")
cells = [(100, 40, 160, 60), (100, 100, 160, 120), (100, 160, 160, 180)]
pred_skip = [cells[1], cells[2]]                     # the model skipped the first cell, in reading order
pred_one_box = [(100, 40, 160, 180)]                 # one box around all three
for name, pred in [("all three, in order", cells), ("first one skipped", pred_skip), ("one box around all", pred_one_box)]:
    m, s = match_f1(pred, cells), sequence_f1(pred, cells)
    R(f"  {name:20s} set-matched F1 {m['f1']:.2f}   position-by-position F1 {s['f1']:.2f}")
    data[f"multi {name}"] = {"set_f1": m["f1"], "sequence_f1": s["f1"]}
R("No target: 'the giraffe' in a kitchen photo")
for name, pred in [("answers 'no such region'", []), ("draws a confident box", [(30, 30, 90, 90)])]:
    R(f"  {name:26s} F1 {match_f1(pred, [])['f1']:.2f}")

# ---- granularity -------------------------------------------------------------------------------
R("")
R("Granularity: the label is one word; the model returns the whole line that contains it")
word = (220, 100, 280, 116)
line = (40, 98, 560, 118)
block = (40, 60, 560, 240)
R(f"  line vs word label: IoU {G.box_iou(line, word):.3f} -> counted wrong, though the evidence is right")
R(f"  block vs word label: IoU {G.box_iou(block, word):.3f}")
nested = all(o[0] <= i[0] and o[1] <= i[1] and o[2] >= i[2] and o[3] >= i[3] for i, o in [(word, line), (line, block)])
R(f"  nested phrase in line in block (what a hierarchy loss enforces): {nested}")

# ---- time --------------------------------------------------------------------------------------
R("")
R("Temporal grounding: one dimension, the same idea")
true_iv = (10.2, 18.0)
for pred in [(11.0, 18.5), (12.0, 19.5), (14.0, 22.0)]:
    t = G.interval_iou(pred, true_iv)
    R(f"  predicted {pred} s vs true {true_iv} s: tIoU {t:.3f}   R@0.5 {'hit' if t >= 0.5 else 'miss'}   "
      f"R@0.7 {'hit' if t >= 0.7 else 'miss'}")
R.save(data)
