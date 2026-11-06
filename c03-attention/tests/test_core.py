"""The hand-written attention against torch's trusted implementation."""

import math

import pytest
import torch
import torch.nn.functional as F

from attnzoo.core import SelfAttentionHead, scaled_dot_product_attention
from attnzoo.masks import causal_mask


def test_shapes_and_rows_sum_to_one():
    q, k, v = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 7, 8), torch.randn(2, 4, 7, 3)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 5, 3)  # (B, H, Nq, dv)
    assert w.shape == (2, 4, 5, 7)  # (B, H, Nq, Nk)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 4, 5))


@pytest.mark.parametrize("causal", [False, True])
def test_manual_matches_torch_sdpa(causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    mask = causal_mask(16) if causal else None
    ours, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(ours, ref, atol=1e-6, rtol=1e-5)


def test_missing_scale_is_detected():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 16, 64) for _ in range(3))
    good, _ = scaled_dot_product_attention(q, k, v)
    bad, w_bad = scaled_dot_product_attention(q, k, v, scale=1.0)  # forgot 1/sqrt(d)
    assert not torch.allclose(good, bad, atol=1e-3)
    # Without the scale, logits have std ~ sqrt(64) = 8, and softmax collapses towards one-hot.
    assert w_bad.amax(-1).mean() > 0.8


def test_single_head_shapes():
    head = SelfAttentionHead(d_model=32, d_head=8)
    assert head(torch.randn(3, 10, 32)).shape == (3, 10, 8)


def test_scale_default_is_inverse_sqrt_d():
    q, k, v = (torch.randn(1, 1, 4, 16) for _ in range(3))
    a, _ = scaled_dot_product_attention(q, k, v)
    b, _ = scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(16))
    torch.testing.assert_close(a, b)
