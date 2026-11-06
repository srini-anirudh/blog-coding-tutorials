"""Every worked example in the post's metrics section, computed by the metric code.

Figure 2 (VQA accuracy), Figure 3 (ANLS), relaxed accuracy, Figure 4 (IoU, Acc@0.5, click
accuracy, gIoU and cIoU), the POPE yes-ratio, CHAIR and Table 1's "Example verdict" column.

    python labs/measuring-what-a-vlm-can-do/scripts/01_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from benchlab import metrics as M  # noqa: E402
from benchlab.report import Report  # noqa: E402
from benchlab.stats import elo_win_prob  # noqa: E402

r = Report("01_metrics")

r("== VQA accuracy (Figure 2): ten humans, four 'red', three 'pink', two 'maroon', one 'orange'")
humans = ["red"] * 4 + ["pink"] * 3 + ["maroon"] * 2 + ["orange"]
for ans in ["red", "pink", "maroon", "orange", "blue"]:
    naive = min(humans.count(ans) / 3, 1)
    r(f"  {ans:7s} official (10 leave-one-out subsets) {M.vqa_accuracy(ans, humans):.2f}   naive min(n/3,1) {naive:.2f}")
r("  normalisation:", {s: M.vqa_normalize(s) for s in ["Two.", "Red.", "the bus", "there are 2 people", "1,250"]})
r("  'Two people.' vs ten '2':", M.vqa_accuracy("Two people.", ["2"] * 10),
  "  'Two.' vs ten '2':", M.vqa_accuracy("Two.", ["2"] * 10))

r("\n== ANLS (Figure 3), tau = 0.5, gold 'SPRINGFIELD'")
for p in ["SPRINGFIELD", "SPRlNGFIELD", "Springfeld", "SPRING", "SHELBYVILLE", ""]:
    nl = M.levenshtein(p.lower(), "springfield") / max(len(p), len("springfield"))
    r(f"  {p!r:15s} edits {M.levenshtein(p.lower(), 'springfield'):2d}  NL {nl:.3f}  ANLS {M.anls(p, 'SPRINGFIELD'):.3f}")
r(f"  numbers: '7,250' vs gold '1,250' -> ANLS {M.anls('7,250', '1,250'):.2f} (one character, a different value)")

r("\n== Relaxed accuracy (ChartQA, 5% relative tolerance)")
for p, g in [("4.19", "4.2"), ("4.5", "4.2"), ("2020", "2019"), ("104", "100"), ("106", "100"), ("Red", "red"), ("2", "two")]:
    r(f"  pred {p:5s} gold {g:5s} -> {M.relaxed_accuracy(p, g):.0f}")
r("  (the year 2020 against 2019 is off by 0.05% and counts as correct)")

r("\n== IoU and Acc@0.5 (Figure 4): the same 18-pixel shift on a large and a small box")
big, small = (100, 100, 400, 400), (100, 100, 130, 130)
shift = lambda b, d: (b[0] + d, b[1], b[2] + d, b[3])  # noqa: E731
for name, b in [("300 px dog", big), ("30 px icon", small)]:
    iou = M.box_iou(shift(b, 18), b)
    r(f"  {name}: IoU {iou:.3f} -> Acc@0.5 {'pass' if iou > 0.5 else 'fail'}")
r(f"  Acc@0.5 over the two: {M.acc_at_iou([shift(big, 18), shift(small, 18)], [big, small]):.2f}")
r(f"  boundary: IoU 0.49 -> {float(0.49 > 0.5):.0f}, IoU 0.51 -> {float(0.51 > 0.5):.0f}")
r(f"  click accuracy (ScreenSpot): point (115, 115) in the icon -> {M.point_in_box((115, 115), small):.0f};"
  f" point (133, 115) -> {M.point_in_box((133, 115), small):.0f}")


def square_mask(size, top, left, side):
    m = torch.zeros(size, size, dtype=torch.bool)
    m[top:top + side, left:left + side] = True
    return m


r("\n== gIoU versus cIoU (Table 1): one large mask with IoU 0.9, one small mask with IoU 0.2")
# Large: gold 100x100, prediction 100x90 inside it -> IoU 0.9. Small: gold 10x10, prediction 10x2 -> 0.2.
g_large = square_mask(256, 0, 0, 100)
p_large = g_large.clone()
p_large[90:100] = False
g_small = square_mask(256, 200, 200, 10)
p_small = torch.zeros_like(g_small)
p_small[200:202, 200:210] = True
for name, p, g in [("large", p_large, g_large), ("small", p_small, g_small)]:
    i, u = M.mask_iou(p, g)
    r(f"  {name}: intersection {i}, union {u}, IoU {i / u:.2f}")
gi, ci = M.giou_ciou([p_large, p_small], [g_large, g_small])
r(f"  gIoU {gi:.2f} (mean of per-image IoU)   cIoU {ci:.3f} (pooled pixels: the large object decides it)")

r("\n== POPE: a model that says 'yes' too often, on a balanced set of 10 questions")
gold = ["yes"] * 5 + ["no"] * 5
for name, preds in [("calibrated", ["yes"] * 4 + ["no"] + ["no"] * 4 + ["yes"]),
                    ("yes-biased", ["yes"] * 5 + ["yes"] * 3 + ["no"] * 2)]:
    s = M.pope_stats(preds, gold)
    r(f"  {name:10s} " + "  ".join(f"{k} {v:.2f}" for k, v in s.items() if k != "n"))
r("  answer rule:", {t: M.yes_no(t) for t in ["Yes.", "No, there is not.", "There is a dog.", "yes there is"]})

r("\n== CHAIR on three captions")
mentioned = [{"dog", "frisbee"}, {"car", "person", "bicycle"}, {"cat"}]
present = [{"dog", "frisbee", "grass"}, {"car", "person"}, {"dog"}]
r("  ", M.chair(mentioned, present), " (2 of 6 mentions absent; 2 of 3 captions hallucinate)")

r("\n== Elo: P(i beats j) for a rating gap")
for gap in [0, 100, 200, 400]:
    r(f"  gap {gap:3d}: {elo_win_prob(1000 + gap, 1000):.3f}")
r(f"  a 400-point gap wins {elo_win_prob(1400, 1000):.4f} = 10/11 = {10 / 11:.4f}")
r.save()
