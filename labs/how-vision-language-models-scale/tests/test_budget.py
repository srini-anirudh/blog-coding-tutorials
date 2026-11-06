import math

import pytest

from vlmscale.budget import (K_FROZEN, K_FROZEN_PASS_GRAD, attention_flops, encoder_share, num_patches,
                             num_visual_tokens, sample_flops)
from vlmscale.ladder import CROSS_CHECKS, LADDER, order_of_magnitude_match

G = 1e9


def test_patch_and_token_counts():
    assert num_patches(448, patch=14) == 1024
    assert num_visual_tokens(1024, 2) == 256
    with pytest.raises(ValueError):
        num_visual_tokens(1023, 2)


def test_figure1_shares():
    assert sample_flops(0.4 * G, 7 * G, 448, 14, 2, 200).encoder_share == pytest.approx(0.114, abs=0.002)
    assert sample_flops(0.4 * G, 0.5 * G, 448, 14, 2, 200).encoder_share == pytest.approx(0.642, abs=0.002)


@pytest.mark.parametrize("n_v,n_l,m,t_over_v", [(0.675, 1.5, 2, 0), (0.4, 7, 2, 200 / 256), (0.093, 0.135, 4, 0.5)])
def test_closed_form_matches_sum(n_v, n_l, m, t_over_v):
    side, p = 448, 14
    v = num_visual_tokens(num_patches(side, patch=p), m) if m < 4 else None
    if v is None:
        side, p = 512, 16
        v = num_visual_tokens(num_patches(side, patch=p), m)
    f = sample_flops(n_v * G, n_l * G, side, p, m, text_tokens=round(t_over_v * v))
    assert encoder_share(n_v, n_l, m, round(t_over_v * v) / v) == pytest.approx(f.encoder_share)


def test_post_numbers():
    assert encoder_share(0.675, 1.5, 2) == pytest.approx(0.643, abs=0.002)
    assert encoder_share(0.675, 72, 2) < 0.04
    assert encoder_share(0.093, 0.135, 2) == pytest.approx(0.73, abs=0.01)
    assert 1 - K_FROZEN_PASS_GRAD / 6 == pytest.approx(1 / 3)
    assert K_FROZEN == 2


def test_resolution_scales_quadratically_and_attention_quartically():
    a, b = sample_flops(1, 1, 448, 14, 2), sample_flops(1, 1, 896, 14, 2)
    assert b.encoder / a.encoder == 4 and b.llm / a.llm == 4
    assert attention_flops(4096, 64, 2) / attention_flops(1024, 64, 2) == 16
    assert attention_flops(4096, 64, 4, window=64, full_layers=0) == pytest.approx(attention_flops(4096, 64, 4) * 64 / 4096)


def test_ladder_matches_post():
    for model, _, _, flops, post in LADDER:
        assert order_of_magnitude_match(flops, post), model
    assert math.isclose(CROSS_CHECKS["Chameleon-34B, 4.28M A100-hours"], LADDER[11][3], rel_tol=0.05)
