"""Sinusoidal, RoPE and context-extension arithmetic."""

import math

import pytest
import torch

from posenc import extend, rope, sinusoidal


def test_sinusoidal_shift_is_a_fixed_rotation():
    p = torch.arange(200)
    for k in (1, 5, 77):
        R = sinusoidal.shift_matrix(k, 32)
        torch.testing.assert_close(sinusoidal.sinusoidal(p + k, 32), sinusoidal.sinusoidal(p, 32) @ R.T)


def test_sinusoidal_dot_depends_only_on_offset():
    p = torch.arange(300)
    dots = (sinusoidal.sinusoidal(p, 32) * sinusoidal.sinusoidal(p + 9, 32)).sum(-1)
    assert float(dots.max() - dots.min()) < 1e-9


def test_rope_score_is_relative_and_norm_preserving():
    torch.manual_seed(0)
    f = rope.inv_freq(32)
    q, k = torch.randn(8, 1, 32), torch.randn(8, 1, 32)

    def score(m, n):
        return (rope.apply_half(q, torch.tensor([m]), f) * rope.apply_half(k, torch.tensor([n]), f)).sum(-1)

    torch.testing.assert_close(score(7, 2), score(107, 102), atol=1e-4, rtol=1e-4)
    rq = rope.apply_half(q, torch.tensor([999]), f)
    torch.testing.assert_close(rq.norm(dim=-1), q.norm(dim=-1))


def test_conventions_differ_by_a_permutation_only():
    torch.manual_seed(0)
    f, P, pos = rope.inv_freq(16), rope.half_to_interleaved(16), torch.arange(10)
    x = torch.randn(2, 10, 16)
    torch.testing.assert_close(rope.apply_interleaved(x[..., P], pos, f), rope.apply_half(x, pos, f)[..., P])
    assert not torch.allclose(rope.apply_interleaved(x, pos, f), rope.apply_half(x, pos, f))


def test_half_split_matches_c03_rotary_embedding():
    from posenc.c03 import RotaryEmbedding, rotate

    emb = RotaryEmbedding(16, 32)
    torch.testing.assert_close(emb.inv_freq, rope.inv_freq(16))
    x, pos = torch.randn(1, 1, 10, 16), torch.arange(10)
    cos, sin = emb(pos[None])
    torch.testing.assert_close(rope.apply_half(x, pos, rope.inv_freq(16)), rotate(x, cos, sin))


def test_decay_envelope_starts_at_its_maximum_and_falls():
    env = rope.decay_envelope(64, 10_000, torch.tensor([0, 10, 100]))
    assert env[0] > env[1] > env[2]


def test_planes_without_full_turn():
    lam = rope.wavelengths(128, 10_000)
    assert math.isclose(float(lam[0]), 2 * math.pi)
    assert rope.planes_without_full_turn(128, 10_000, 10**9) == 0
    assert rope.planes_without_full_turn(128, 1e6, 4096) > rope.planes_without_full_turn(128, 1e4, 4096)


def test_bf16_table_loses_angle_at_long_positions():
    f = rope.inv_freq(128, 500_000, torch.float64)
    pos = torch.tensor([500_000])
    assert float(rope.wrapped_angle_error(pos, f, torch.float32).median()) < 1e-3
    assert float(rope.wrapped_angle_error(pos, f, torch.bfloat16).median()) > 0.1


def test_pi_and_ntk_speeds():
    theta = rope.inv_freq(32)
    pi, _ = extend.scaled_inv_freq("pi", 32, 10_000, 4, 64)
    ntk, _ = extend.scaled_inv_freq("ntk", 32, 10_000, 4, 64)
    torch.testing.assert_close(pi, theta / 4)
    torch.testing.assert_close(ntk[0], theta[0])  # fastest plane untouched
    torch.testing.assert_close(ntk[-1], theta[-1] / 4)  # slowest plane fully interpolated
    assert extend.out_of_range_planes(pi, theta, 64, 256) == 0
    assert extend.out_of_range_planes(theta, theta, 64, 256) > 0


@pytest.mark.parametrize("head_dim,base,factor,orig", [(32, 10_000, 4, 64), (128, 500_000, 8, 8192)])
def test_yarn_matches_transformers(head_dim, base, factor, orig):
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    cfg = LlamaConfig(hidden_size=head_dim * 2, num_attention_heads=2, head_dim=head_dim,
                      max_position_embeddings=orig * factor,
                      rope_parameters={"rope_type": "yarn", "rope_theta": base, "factor": float(factor),
                                       "original_max_position_embeddings": orig})
    ref, attn_factor = ROPE_INIT_FUNCTIONS["yarn"](cfg, "cpu")
    ours, scale = extend.scaled_inv_freq("yarn", head_dim, base, factor, orig)
    torch.testing.assert_close(ours, ref.float())
    assert math.isclose(scale, attn_factor**2, rel_tol=1e-6)  # HF scales cos and sin, so q.k gets the square
