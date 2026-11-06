import torch

from twomod import masks
from twomod.positions import (
    alignment_score,
    interpolate_table,
    neighbour_score,
    sincos_2d,
    sinusoidal,
    sinusoidal_shift_matrix,
)


def test_sinusoidal_offset_is_a_fixed_linear_map():
    table = sinusoidal(200, 32)
    for k in (1, 5, 50):
        torch.testing.assert_close(table[k:], table[:-k] @ sinusoidal_shift_matrix(k, 32), atol=1e-5, rtol=0)


def test_neighbour_score_separates_random_from_2d_tables():
    torch.manual_seed(0)
    assert neighbour_score(torch.randn(64, 64), 8, 8) < 0.2
    assert neighbour_score(sincos_2d(8, 8, 64), 8, 8) > 0.9


def test_interpolation_keeps_layout_and_raster_reuse_breaks_it():
    old = sincos_2d(6, 6, 64)
    new = interpolate_table(old, (6, 6), (10, 10))
    assert new.shape == (100, 64)
    assert alignment_score(old, new, (6, 6), (10, 10)) > 0.9
    assert alignment_score(old, old.repeat(3, 1)[:100], (6, 6), (10, 10)) < 0.3
    torch.testing.assert_close(interpolate_table(old, (6, 6), (6, 6)), old, atol=1e-5, rtol=0)


def test_masks():
    assert torch.equal(masks.causal(4), torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    assert masks.bidirectional(3).all()
    m = masks.prefix_lm(6, 3)
    assert m[:3, :3].all() and not m[:3, 3:].any() and torch.equal(m[3:, 3:], masks.causal(3))
    padded = masks.with_padding(masks.causal(4), torch.tensor([[1, 1, 0, 0]]))
    assert not padded[0, 3, 2] and padded[0, 3, 3] and padded[0, 1, 0]
    assert masks.render(masks.causal(2)) == "#.\n##"
