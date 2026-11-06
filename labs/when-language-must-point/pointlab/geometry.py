"""Regions as pixels: rasterise shapes, take their tightest box, and compare them with IoU.

Every shape is a polygon in pixel coordinates. A pixel belongs to the shape when its centre lies inside
the polygon, which is the same rule the post's interactive figures use.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.path import Path as MplPath

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2), pixel units, x2 and y2 exclusive


# ---- rasterising -----------------------------------------------------------------------------

def polygon_mask(pts, width: int, height: int) -> np.ndarray:
    """Boolean (H, W) mask of the pixels whose centres fall inside the polygon."""
    ys, xs = np.mgrid[0:height, 0:width]
    centres = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
    inside = MplPath(np.asarray(pts, dtype=float)).contains_points(centres)
    return inside.reshape(height, width)


def box_mask(box: Box, width: int, height: int) -> np.ndarray:
    """Pixels whose centres fall inside the box."""
    ys, xs = np.mgrid[0:height, 0:width] + 0.5
    x1, y1, x2, y2 = box
    return (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)


# ---- the shapes from the post ----------------------------------------------------------------

def annular_sector(cx, cy, r1, r2, a0, a1, n=64):
    """A curved band, like a line of text bent around a seal. Angles in degrees, counter-clockwise."""
    def p(r, a):
        return (cx + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a)))
    outer = [p(r2, a1 - (a1 - a0) * k / (n - 1)) for k in range(n)]
    inner = [p(r1, a0 + (a1 - a0) * k / (n - 1)) for k in range(n)]
    return outer + inner


def figure2_band(n=64):
    """The curved line of text in the post's Figure 2 (600 x 330 canvas)."""
    return annular_sector(300, 272, 150, 190, 20, 160, n)


def blob(n=120):
    """Figure 9's compact object: a slightly wobbly ellipse."""
    pts = []
    for k in range(n):
        t = k / n * 2 * math.pi
        r = 1 + 0.09 * math.sin(3 * t) + 0.05 * math.cos(5 * t)
        pts.append((300 + 118 * r * math.cos(t), 165 + 84 * r * math.sin(t)))
    return pts


def wedge(cx=190, cy=285, r=250, a0=-80, a1=-22, n=40):
    """Figure 9's pie wedge (angles in screen coordinates, y pointing down)."""
    pts = [(cx, cy)]
    for k in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * k / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def figure9_band():
    """Figure 9's curved text line."""
    cx, cy, r1, r2 = 300, 330, 150, 186
    out = [(cx + r2 * math.cos(math.radians(155 - 130 * k / 40)), cy - r2 * math.sin(math.radians(155 - 130 * k / 40)))
           for k in range(41)]
    inn = [(cx + r1 * math.cos(math.radians(25 + 130 * k / 40)), cy - r1 * math.sin(math.radians(25 + 130 * k / 40)))
           for k in range(41)]
    return out + inn


def rotated_bar(cx, cy, length, thickness, degrees):
    """A thin straight text line, rotated about its centre."""
    a = math.radians(degrees)
    ca, sa = math.cos(a), math.sin(a)
    corners = [(-length / 2, -thickness / 2), (length / 2, -thickness / 2),
               (length / 2, thickness / 2), (-length / 2, thickness / 2)]
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in corners]


def figure9_diagonal():
    return rotated_bar(300, 165, 440, 24, -27)


def ellipse(cx, cy, rx, ry, n=256):
    return [(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n)]


# ---- boxes, points and IoU -------------------------------------------------------------------

def tight_box(mask: np.ndarray) -> Box | None:
    """Tightest pixel box around the foreground."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def fill_ratio(mask: np.ndarray) -> float:
    """f(S) = |S| / |B*|: the share of the tightest box that the region fills."""
    x1, y1, x2, y2 = tight_box(mask)
    return float(mask.sum() / ((x2 - x1) * (y2 - y1)))


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean() + 0.5), float(ys.mean() + 0.5)


def point_in_mask(pt, mask: np.ndarray, tolerance: float = 0.0) -> bool:
    """Inside test. With a tolerance, a point within that many pixels of the region also counts (pointing game)."""
    x, y = pt
    if tolerance <= 0:
        xi, yi = int(math.floor(x)), int(math.floor(y))
        return 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and bool(mask[yi, xi])
    ys, xs = np.nonzero(mask)
    return bool((((xs + 0.5 - x) ** 2 + (ys + 0.5 - y) ** 2) <= tolerance ** 2).any())


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """One-dimensional IoU: used for one box edge in Figure 5 and for time intervals in video."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0
