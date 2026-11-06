import random

import numpy as np
import pytest

from vlmtrain import grounding as G
from vlmtrain.metrics import box_iou
from vlmtrain.synth.shapes import Obj, make_scene, object_mask


def test_box_formats_carry_the_same_bins():
    box = (14.0, 70.0, 63.0, 112.0)
    tok = G.encode_box(box, 224, 224, "tokens")
    txt = G.encode_box(box, 224, 224, "text")
    assert tok == "<loc_2><loc_10><loc_8><loc_15>"
    assert txt == "[2, 10, 8, 15]"
    assert G.parse_coords(tok, "tokens") == G.parse_coords(txt, "text")


@pytest.mark.parametrize("fmt", ["tokens", "text"])
def test_box_roundtrip_loses_at_most_one_bin(fmt):
    rng = random.Random(0)
    for _ in range(200):
        x0, y0 = rng.uniform(0, 160), rng.uniform(0, 160)
        box = (x0, y0, x0 + rng.uniform(28, 56), y0 + rng.uniform(28, 56))
        dec = G.decode_box(G.encode_box(box, 224, 224, fmt), 224, 224, fmt)
        assert all(abs(a - b) <= 224 / G.LOC_BINS for a, b in zip(dec, box))
        assert box_iou(dec, box) > 0.45  # the bin grid itself caps IoU below 1


def test_bin_aligned_box_is_exact():
    s = 224 / G.LOC_BINS
    box = (2 * s, 3 * s, 6 * s, 9 * s)
    assert G.decode_box(G.encode_box(box, 224, 224), 224, 224) == pytest.approx(box)


def test_malformed_output_does_not_parse():
    assert G.decode_box("<loc_3><loc_4>", 224, 224) is None
    assert G.decode_box("the red circle", 224, 224, "text") is None
    assert G.decode_point("[5, 9]", 224, 224, "text") == pytest.approx((5.5 * 7, 9.5 * 7))


def test_rle_roundtrip_and_coco_convention():
    m = np.zeros((4, 3), dtype=bool)
    m[0, 0] = True
    rle = G.mask_to_rle(m)
    assert rle["counts"][0] == 0  # starts with foreground -> leading zero count
    assert (G.rle_to_mask(rle) == m).all()
    _, objs = make_scene(random.Random(3), 3, 3)
    mask = object_mask(objs[0], 224)
    assert (G.rle_to_mask(G.mask_to_rle(mask)) == mask).all()


def test_mask_box_and_iou():
    obj = Obj("square", "red", (10.0, 20.0, 40.0, 50.0))
    mask = object_mask(obj, 224)
    assert G.mask_to_box(mask) == (10.0, 20.0, 40.0, 50.0)
    assert G.mask_iou(mask, mask) == 1.0
    assert G.mask_iou(mask, ~mask) == 0.0


def test_polygon_encoding():
    obj = Obj("triangle", "blue", (0.0, 0.0, 70.0, 70.0))
    s = G.encode_polygon(obj.polygon(), 224, 224, "text")
    assert G.parse_coords(s, "text") == [5, 0, 10, 10, 0, 10]


@pytest.mark.parametrize("shape", ["circle", "square", "triangle"])
def test_polygon_target_rasterises_close_to_the_true_mask(shape):
    obj = Obj(shape, "red", (40.0, 60.0, 96.0, 116.0))
    text = G.encode_polygon(obj.polygon(), 224, 224)
    points = G.decode_polygon(text, 224, 224)
    assert G.mask_iou(G.polygon_to_mask(points, 224, 224), object_mask(obj, 224)) > 0.7  # the bin grid caps it


def test_short_polygon_does_not_parse():
    assert G.decode_polygon("<loc_1><loc_2><loc_3><loc_4>", 224, 224) is None
