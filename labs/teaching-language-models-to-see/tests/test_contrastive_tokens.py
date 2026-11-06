"""Contrastive losses and visual-token arithmetic. No downloads."""

import math

import pytest
import torch
import torch.nn.functional as F

from seelab.contrastive import (
    ToyPairs, clip_loss, clip_loss_chunked_wrong, modality_gap, siglip_loss, siglip_loss_chunked, train_toy,
)
from seelab.tokens import (
    adam_bytes, ceil_grid_tokens, internvl_grid, loss_share, native_tokens, smart_resize, square_tokens, tiled_tokens,
)


def pair(n=24, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = F.normalize(torch.randn(n, d, generator=g, dtype=torch.float64), dim=-1)
    v = F.normalize(u + 0.5 * torch.randn(n, d, generator=g, dtype=torch.float64), dim=-1)
    return u, v


@pytest.mark.parametrize("chunk", [1, 5, 8, 24])
def test_sigmoid_loss_splits_into_blocks(chunk):
    u, v = pair()
    assert torch.allclose(siglip_loss(u, v, 10.0, -10.0), siglip_loss_chunked(u, v, 10.0, -10.0, chunk), atol=1e-12)


def test_softmax_loss_does_not_split():
    u, v = pair()
    assert not torch.allclose(clip_loss(u, v, 0.1), clip_loss_chunked_wrong(u, v, 0.1, 8))
    assert torch.allclose(clip_loss(u, v, 0.1), clip_loss_chunked_wrong(u, v, 0.1, 24))  # one block = the full loss


def test_clip_loss_matches_the_formula():
    u, v = pair(n=6)
    s = u @ v.T / 0.07
    manual = -sum(math.log(s[i].softmax(0)[i]) + math.log(s[:, i].softmax(0)[i]) for i in range(6)) / 12
    assert clip_loss(u, v, 0.07).item() == pytest.approx(manual, rel=1e-9)


def test_siglip_loss_matches_the_formula():
    u, v = pair(n=5)
    s = u @ v.T
    manual = -sum(math.log(torch.sigmoid((1 if i == j else -1) * (10 * s[i, j] - 10))) for i in range(5) for j in range(5)) / 5
    assert siglip_loss(u, v, 10.0, -10.0).item() == pytest.approx(manual, rel=1e-9)


def test_modality_gap_is_zero_for_identical_sets():
    u, _ = pair()
    r = modality_gap(u, u)
    assert r["gap"] == pytest.approx(0, abs=1e-6) and r["cos_matched"] == pytest.approx(1)


def test_toy_training_lowers_the_loss():
    data = ToyPairs()
    _, hist = train_toy("siglip", 32, 60, data, log_every=59)
    assert hist[-1]["loss"] < hist[0]["loss"]


def test_token_counts_quoted_in_the_post():
    assert square_tokens(336, 14) == 576
    assert square_tokens(448, 14, 2) == 256
    assert square_tokens(896, 14) == 16 * square_tokens(224, 14)
    assert tiled_tokens(1000, 1300, max_num=6) == (2, 3, 1792)
    assert native_tokens(1000, 1300) == (36, 46, 1656)
    assert ceil_grid_tokens(1000, 1300, 14, 2) == (36, 47, 1692)  # Figure 7's rounding gives a different count


def test_smart_resize_rounds_to_multiples_and_respects_the_cap():
    h, w = smart_resize(1300, 1000)
    assert (h, w) == (1288, 1008)
    h, w = smart_resize(30000, 30000)
    assert h % 28 == 0 and w % 28 == 0 and h * w <= 14 * 14 * 4 * 16384


def test_internvl_grid_follows_the_aspect_ratio():
    assert internvl_grid(1000, 1300, max_num=12) == (3, 4)
    assert internvl_grid(1920, 1080, max_num=6) == (2, 1)
    assert internvl_grid(448, 448, max_num=12) == (1, 1)


def test_loss_share_and_memory_rule():
    assert loss_share([("image", 576, False), ("text", 12, True)]) == (12, 588, 12 / 588)
    assert adam_bytes(7e9, 0) == 112e9 and adam_bytes(0, 7e9) == 14e9
