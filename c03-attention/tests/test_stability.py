"""QK-norm, soft-capping, biases and the research variants."""

import pytest
import torch

from attnzoo.biases import RelativePositionBias, alibi_bias, alibi_slopes
from attnzoo.config import AttentionConfig
from attnzoo.gqa import GroupedQueryAttention
from attnzoo.heads import merge_heads
from attnzoo.stability import RMSNorm, logit_stats, soft_cap
from attnzoo.variants import CosineAttention, DifferentialAttention, GatedAttention, TalkingHeadsAttention


def test_soft_cap_bounds_and_is_near_identity_for_small_logits():
    s = torch.linspace(-200, 200, 1001)
    capped = soft_cap(s, 30.0)
    assert capped.abs().max() < 30.0
    small = torch.linspace(-1, 1, 11)
    torch.testing.assert_close(soft_cap(small, 30.0), small, atol=1e-3, rtol=0)


def test_qk_norm_bounds_logits_when_activations_grow():
    norm = RMSNorm(16)
    for growth in (1.0, 100.0):
        q, k = torch.randn(1, 1, 32, 16) * growth, torch.randn(1, 1, 32, 16) * growth
        raw = (q @ k.transpose(-2, -1)) / 4
        normed = (norm(q) @ norm(k).transpose(-2, -1)) / 4
        assert normed.abs().max() <= 16 / 4 + 1e-4  # |q.k| <= ||q|| ||k|| = d after RMS norm
    assert logit_stats(raw)["max_logit"] > 1000  # without the norm, logits scale with growth^2


def test_alibi_slopes():
    torch.testing.assert_close(alibi_slopes(8), torch.tensor([2.0 ** -(i + 1) for i in range(8)]))
    assert len(alibi_slopes(12)) == 12
    b = alibi_bias(alibi_slopes(4), 5, 5)
    assert b.shape == (1, 4, 5, 5) and b[0, 0, 4, 4] == 0 and b[0, 0, 4, 0] == -0.25 * 4


def test_relative_position_bias_shape():
    assert RelativePositionBias(4, 8)(5, 9).shape == (1, 4, 5, 9)


@pytest.mark.parametrize("pos", ["alibi", "none"])
def test_gqa_with_alibi_or_none_and_qk_norm(pos):
    m = GroupedQueryAttention(AttentionConfig("gqa", 32, 4, num_kv_heads=2, positional_encoding=pos, qk_norm=True))
    assert m(torch.randn(2, 9, 32)).shape == (2, 9, 32)


def test_softcap_needs_naive_backend():
    x = torch.randn(1, 5, 32)
    m = GroupedQueryAttention(AttentionConfig("gqa", 32, 4, softcap=20.0, backend="naive"))
    assert m(x).shape == (1, 5, 32)
    m.config.backend = "sdpa"
    with pytest.raises(ValueError):
        m(x)


@pytest.mark.parametrize("cls", [GatedAttention, CosineAttention, DifferentialAttention, TalkingHeadsAttention])
def test_variants_shapes_causality_and_grads(cls):
    torch.manual_seed(0)
    m = cls(AttentionConfig("mha", 32, 4))
    x = torch.randn(1, 8, 32, requires_grad=True)
    y = x.detach().clone()
    y[:, 5:] = 0
    out = m(x)
    assert out.shape == (1, 8, 32)
    torch.testing.assert_close(out[:, :5], m(y)[:, :5])
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_talking_heads_at_init_is_mha():
    """With identity mixing matrices, talking-heads reduces to plain causal MHA."""
    torch.manual_seed(0)
    th = TalkingHeadsAttention(AttentionConfig("mha", 32, 4))
    x = torch.randn(1, 6, 32)
    q, k, v = th.project(x)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(th(x), th.o_proj(merge_heads(ref)), atol=1e-5, rtol=1e-4)
