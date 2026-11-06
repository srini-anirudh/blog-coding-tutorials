import math

import numpy as np
import pytest

from pointlab import geometry as G


def test_best_box_iou_is_fill_ratio():
    rng = np.random.default_rng(0)
    for _ in range(20):
        pts = rng.uniform(10, 190, (6, 2))
        m = G.polygon_mask(pts, 200, 200)
        if m.sum() < 30:
            continue
        box = G.tight_box(m)
        assert G.mask_iou(G.box_mask(box, 200, 200), m) == pytest.approx(G.fill_ratio(m))


def test_ellipse_fill_is_pi_over_four():
    m = G.polygon_mask(G.ellipse(300, 165, 200, 110, n=720), 600, 330)
    assert G.fill_ratio(m) == pytest.approx(math.pi / 4, abs=0.01)


def test_curved_band_fill_and_centroid():
    m = G.polygon_mask(G.figure2_band(256), 600, 330)
    assert G.fill_ratio(m) == pytest.approx(0.34, abs=0.01)
    assert not G.point_in_mask(G.centroid(m), m)       # the centroid falls in the hollow
    assert G.point_in_mask((300, 102), m)


def test_thin_diagonal_below_point_one():
    m = G.polygon_mask(G.rotated_bar(500, 500, 700, 35, 45), 1000, 1000)
    assert G.fill_ratio(m) < 0.1


def test_box_and_interval_iou():
    assert G.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert G.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert G.box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert G.interval_iou((268, 884), (268, 584)) == pytest.approx(316 / 616)


def test_point_tolerance():
    m = np.zeros((50, 50), bool)
    m[20:30, 20:30] = True
    assert not G.point_in_mask((10.0, 25.0), m)
    assert G.point_in_mask((10.0, 25.0), m, tolerance=15)
