"""Point, box, polygon, mask: the best box has IoU equal to the fill ratio (post Figures 2 and 9)."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointlab import geometry as G  # noqa: E402
from pointlab.coords import digit_split_count  # noqa: E402
from pointlab.report import Report  # noqa: E402

R = Report("01_shapes")
W, H = 600, 330

# ---- Figure 2: one curved line of text, five answers ------------------------------------------
cx, cy, r1, r2, a0, a1 = 300, 272, 150, 190, 20, 160
ys, xs = np.mgrid[0:H, 0:W] + 0.5
rr = np.hypot(xs - cx, cy - ys)
ang = np.degrees(np.arctan2(cy - ys, xs - cx))
band = (rr >= r1) & (rr <= r2) & (ang >= a0) & (ang <= a1)          # the exact region, as in the post

box = G.tight_box(band)
f = G.fill_ratio(band)
R("Figure 2: a curved band (annular sector r = 150..190 px, 20..160 degrees) on a 600 x 330 canvas")
R(f"  area {band.sum()} px, tightest box {box}, fill ratio f = {f:.3f}")
R(f"  IoU(tightest box, band) = {G.mask_iou(G.box_mask(box, W, H), band):.3f}   (equals f, as derived in the post)")

al = math.radians((a1 - a0) / 2)
rc = (2 / 3) * (r2 ** 3 - r1 ** 3) / (r2 ** 2 - r1 ** 2) * math.sin(al) / al
cen_formula = (cx, cy - rc)
cen_pixels = G.centroid(band)
on_band = (cx, cy - (r1 + r2) / 2)
R(f"  centroid (formula) {cen_formula[0]:.1f}, {cen_formula[1]:.1f}; from pixels {cen_pixels[0]:.1f}, {cen_pixels[1]:.1f}")
R(f"  centroid inside the band? {G.point_in_mask(cen_pixels, band)}.  A point on the band ({on_band[0]}, {on_band[1]}) "
  f"inside? {G.point_in_mask(on_band, band)}")


def cells_mask(cell=6):
    """The figure's 'mask': the band drawn as 6-pixel cells, as a decoder at coarse resolution would."""
    m = np.zeros_like(band)
    for y0 in range(78, 228, cell):
        for x0 in range(114, 488, cell):
            if band[y0 + cell // 2, x0 + cell // 2]:
                m[y0:y0 + cell, x0:x0 + cell] = True
    return m


def coords_text(pts):
    return "[" + ",".join(f"({int(round(x)):03d},{int(round(y)):03d})" for x, y in pts) + "]"


rows = [["point (centroid)", 2, digit_split_count(coords_text([cen_pixels])), "miss (outside)"],
        ["point (on band)", 2, digit_split_count(coords_text([on_band])), "hit"],
        ["box", 4, digit_split_count(coords_text([box[:2], box[2:]])), f"{G.mask_iou(G.box_mask(box, W, H), band):.2f}"]]
poly_iou = {}
for k in (3, 8, 24):
    pts = G.annular_sector(cx, cy, r1, r2, a0, a1, n=k)
    poly_iou[2 * k] = G.mask_iou(G.polygon_mask(pts, W, H), band)
    rows.append([f"polygon, {2 * k} vertices", 4 * k, digit_split_count(coords_text(pts)), f"{poly_iou[2 * k]:.2f}"])
rows.append(["mask (6 px cells, from a decoder)", "-", "1 (<SEG>)", f"{G.mask_iou(cells_mask(), band):.2f}"])
R("")
R.table(["representation", "numbers", "text tokens (3-digit coords, digits split)", "IoU with the band"], rows)

# ---- Fill ratios of common shapes -------------------------------------------------------------
R("")
R("Fill ratio f(S) = |S| / |B*| = the IoU of the best possible box")
fr_rows = []
shapes = {"rectangle": [(100, 50), (500, 50), (500, 280), (100, 280)],
          "ellipse": G.ellipse(300, 165, 200, 110, n=720),
          "Figure 9 blob": G.blob(), "Figure 9 pie wedge": G.wedge(), "Figure 9 curved text line": G.figure9_band(),
          "Figure 9 diagonal line (-27 deg, 440 x 24)": G.figure9_diagonal()}
fill = {}
for name, pts in shapes.items():
    m = G.polygon_mask(pts, W, H)
    tb = G.tight_box(m)
    fill[name] = G.fill_ratio(m)
    fr_rows.append([name, f"{fill[name]:.3f}", f"{G.mask_iou(G.box_mask(tb, W, H), m):.3f}"])
fr_rows.append(["ellipse, exact value pi / 4", f"{math.pi / 4:.3f}", "-"])
R.table(["shape", "fill ratio", "IoU(tightest box, shape)"], fr_rows)

R("")
R("Figure 9, 'Snap to tightest box': box IoU against the true box is 1.00 for every shape; mask IoU is the fill ratio.")
for name in [n for n in shapes if n.startswith("Figure 9")]:
    R(f"  {name:44s} box IoU 1.00 (pass)   mask IoU {fill[name]:.2f} ({'pass' if fill[name] >= 0.5 else 'fail'} at 0.5)")

# ---- Thin text lines at 45 degrees ------------------------------------------------------------
R("")
R("A straight text line of length L and thickness t at 45 degrees: f = 2 L t / (L + t)^2")
for ratio in (0.2, 0.1, 0.05, 0.03):
    L, t = 1000.0, 1000.0 * ratio
    m = G.polygon_mask(G.rotated_bar(500, 500, L / 1000 * 700, t / 1000 * 700, 45), 1000, 1000)
    R(f"  t / L = {ratio:<5} formula {2 * L * t / (L + t) ** 2:.3f}   rasterised {G.fill_ratio(m):.3f}")

# ---- Masks as text: a Text4Seg-style grid with run-length coding --------------------------------
R("")
grid = 16
gy = ((np.arange(grid) + 0.5) * H / grid).astype(int)
gx = ((np.arange(grid) + 0.5) * W / grid).astype(int)
labels = band[np.ix_(gy, gx)]
plain = [("text" if v else "others") for row in labels for v in row]
rle = []
for row in labels:
    runs, prev, n = [], None, 0
    for v in row:
        if v == prev:
            n += 1
        else:
            if prev is not None:
                runs.append(f"{'text' if prev else 'others'}*{n}")
            prev, n = v, 1
    runs.append(f"{'text' if prev else 'others'}*{n}")
    rle.append("|".join(runs))
t_plain = len(plain) + (len(plain) - 1)        # one word plus one separator per cell
t_rle = sum(digit_split_count(r) for r in rle) + grid - 1
R(f"Mask as text on a {grid} x {grid} grid (Text4Seg idea): one word per cell = {t_plain} tokens, "
  f"row-wise run-length = {t_rle} tokens ({1 - t_rle / t_plain:.0%} shorter on this one band).")
R("  Text4Seg reports 74% over its datasets; this lab did not reproduce that figure.")
R("  the first rows that touch the band:", " / ".join(r for r in rle if "text" in r)[:120], "...")

R.save({"figure2": {"fill_ratio": f, "box_iou": f, "polygon_iou": poly_iou,
                    "centroid_inside": G.point_in_mask(cen_pixels, band), "mask_cells_iou": G.mask_iou(cells_mask(), band)},
        "fill_ratio": fill})
