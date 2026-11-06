"""Cross-attention, Q-Former, Perceiver resampler and fusion patterns."""

import pytest
import torch

from attnzoo.config import AttentionConfig
from attnzoo.cross import CrossAttention, GatedCrossAttentionBlock
from attnzoo.fusion import CrossAttentionFusionBlock, EarlyFusionBlock, early_fusion_sequence
from attnzoo.resampler import PerceiverResampler, QFormer


def cfg(**kw):
    return AttentionConfig("cross", 32, 4, causal=False, positional_encoding="none", **kw)


@pytest.mark.parametrize("ns,dc", [(7, 32), (196, 48)])  # text -> text, text -> image
def test_cross_attention_shapes(ns, dc):
    m = CrossAttention(cfg(context_size=dc))
    assert m(torch.randn(2, 5, 32), torch.randn(2, ns, dc)).shape == (2, 5, 32)


def test_cross_attention_ignores_padded_context():
    torch.manual_seed(0)
    m = CrossAttention(cfg()).eval()
    x, ctx = torch.randn(1, 5, 32), torch.randn(1, 6, 32)
    padded = torch.cat([ctx, torch.randn(1, 3, 32)], 1)
    mask = torch.tensor([[1] * 6 + [0] * 3])
    torch.testing.assert_close(m(x, padded, mask), m(x, ctx), atol=1e-5, rtol=1e-4)


def test_gated_block_is_identity_at_init():
    block = GatedCrossAttentionBlock(cfg())
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(block(x, torch.randn(2, 9, 32)), x)


@pytest.mark.parametrize("cls", [QFormer, PerceiverResampler])
def test_resamplers_output_m_tokens_for_any_image_length(cls):
    m = cls(8, hidden_size=32, vision_size=48)
    for nv in (16, 196, 729):
        assert m(torch.randn(2, nv, 48)).shape == (2, 8, 32)  # M = 8 << Nv


@pytest.mark.parametrize("cls", [QFormer, PerceiverResampler])
def test_resamplers_respect_image_padding(cls):
    torch.manual_seed(0)
    m = cls(4, hidden_size=32, vision_size=32).eval()
    img = torch.randn(1, 10, 32)
    padded = torch.cat([img, torch.randn(1, 5, 32)], 1)
    mask = torch.tensor([[1] * 10 + [0] * 5])
    torch.testing.assert_close(m(padded, mask), m(img), atol=1e-5, rtol=1e-4)


def test_early_fusion_mask_and_block():
    seq, mask = early_fusion_sequence(torch.randn(1, 4, 32), torch.randn(1, 3, 32))
    assert seq.shape == (1, 7, 32) and mask.shape == (1, 1, 7, 7)
    assert mask[0, 0, 0, 3] and mask[0, 0, 6, :7].all() and not mask[0, 0, 4, 5]
    out = EarlyFusionBlock(AttentionConfig("gqa", 32, 4))(torch.randn(2, 4, 32), torch.randn(2, 3, 32))
    assert out.shape == (2, 7, 32)


def test_cross_attention_fusion_keeps_text_length():
    block = CrossAttentionFusionBlock(AttentionConfig("gqa", 32, 4))
    assert block(torch.randn(2, 6, 32), torch.randn(2, 50, 32)).shape == (2, 6, 32)
