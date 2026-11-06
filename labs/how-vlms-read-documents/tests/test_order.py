import random

import pytest

from docread.order import (FIG7_BLOCKS, FIG7_REFERENCE, drop_margin_furniture, order_score, raster_order,
                           synthetic_page, train_orderer, xy_cut)


def test_figure7_scores():
    assert order_score(raster_order(FIG7_BLOCKS), FIG7_REFERENCE) == pytest.approx(8 / 13)
    assert order_score(xy_cut(FIG7_BLOCKS), FIG7_REFERENCE) == pytest.approx(2 / 13)
    assert xy_cut(drop_margin_furniture(FIG7_BLOCKS, 450)) == FIG7_REFERENCE


def test_xy_cut_solves_clean_columns_and_breaks_on_loose_boxes():
    rng = random.Random(3)
    clean = [synthetic_page("3-col", rng) for _ in range(20)]
    assert sum(xy_cut(drop_margin_furniture(b, 1400)) == r for b, r in clean) >= 19
    loose = [synthetic_page("2-col, boxes cross the gutter", rng) for _ in range(20)]
    assert sum(xy_cut(drop_margin_furniture(b, 1400)) == r for b, r in loose) <= 10


def test_reference_excludes_furniture():
    blocks, ref = synthetic_page("2-col", random.Random(0))
    furniture = {b.name for b in blocks if b.furniture}
    assert len(furniture) == 2 and not furniture & set(ref)
    assert len(ref) == len(blocks) - 2


def test_learned_orderer_beats_top_edge_sort():
    rng = random.Random(0)
    model = train_orderer([synthetic_page("2-col", rng) for _ in range(60)], steps=300)
    pages = [synthetic_page("2-col", rng) for _ in range(20)]
    learned = sum(order_score(model.order(drop_margin_furniture(b, 1400), 1000, 1400), r) for b, r in pages)
    raster = sum(order_score(raster_order(drop_margin_furniture(b, 1400)), r) for b, r in pages)
    assert learned < raster
