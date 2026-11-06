import pytest
import torch
from PIL import Image

from vlmtrain import resolution as R


@pytest.mark.parametrize("w,h,grid", [(224, 224, (1, 1)), (448, 448, (2, 2)), (448, 224, (1, 2)), (224, 448, (2, 1)),
                                      (448, 160, (1, 3)), (336, 224, (1, 2))])
def test_choose_grid(w, h, grid):
    assert R.choose_grid(w, h, 224, 4) == grid


def test_grid_respects_the_budget():
    r, c = R.choose_grid(4000, 200, 224, 4)
    assert r * c <= 4


@pytest.mark.parametrize("strategy,count", [("resize", 1), ("pad", 1), ("tile", 5)])
def test_tile_counts_for_a_document(strategy, count):
    img = Image.new("RGB", (448, 448), "white")
    x = R.preprocess(img, strategy)
    assert x.shape == (count, 3, 224, 224)
    assert R.n_tiles(448, 448, strategy) == count


def test_normalisation_range():
    assert R.to_tensor(Image.new("RGB", (4, 4), (255, 255, 255))).max() == 1.0
    assert R.to_tensor(Image.new("RGB", (4, 4), (0, 0, 0))).min() == -1.0


def test_tiles_cut_in_row_major_order():
    img = Image.new("RGB", (448, 448), (0, 0, 0))
    img.paste((255, 0, 0), (224, 0, 448, 224))  # top right tile is red
    x = R.preprocess(img, "tile")  # thumbnail, TL, TR, BL, BR
    assert x[2, 0].mean() == pytest.approx(1.0, abs=1e-3)
    assert x[1, 0].mean() == pytest.approx(-1.0, abs=1e-3)


def test_pad_keeps_aspect_ratio():
    img = Image.new("RGB", (448, 224), (0, 0, 0))
    t = R.tiles_for(img, "pad")[0]
    assert t.size == (224, 224)
    assert t.getpixel((112, 2)) == (255, 255, 255)  # padding band on top
    assert t.getpixel((112, 112)) == (0, 0, 0)
    assert torch.is_tensor(R.to_tensor(t))
