"""MHA, fused QKV, MQA and GQA: one family, checked against each other and against SDPA."""

import pytest
import torch
import torch.nn.functional as F

from attnzoo.config import AttentionConfig
from attnzoo.gqa import GroupedQueryAttention
from attnzoo.heads import merge_heads, repeat_kv, split_heads
from attnzoo.mha import FusedQKVAttention, MultiHeadAttention, MultiQueryAttention


def copy_projections(src, dst):
    with torch.no_grad():
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            getattr(dst, name).weight.copy_(getattr(src, name).weight)


def test_split_merge_roundtrip():
    x = torch.randn(2, 5, 32)
    h = split_heads(x, 4)
    assert h.shape == (2, 4, 5, 8)
    torch.testing.assert_close(merge_heads(h), x)


def test_merge_without_contiguous_fails():
    out = torch.randn(2, 4, 5, 8)  # attention output: freshly allocated, contiguous as (B, H, N, d)
    with pytest.raises(RuntimeError):
        out.transpose(1, 2).view(2, 5, 32)  # the bug merge_heads avoids
    assert merge_heads(out).shape == (2, 5, 32)


def test_repeat_kv_maps_query_head_to_group():
    kv = torch.arange(2.0).view(1, 2, 1, 1)  # two KV heads holding 0 and 1
    assert repeat_kv(kv, 3).flatten().tolist() == [0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize("causal", [False, True])
def test_manual_mha_matches_sdpa_backend(causal):
    torch.manual_seed(0)
    naive = MultiHeadAttention(64, 8, causal=causal, backend="naive")
    sdpa = MultiHeadAttention(64, 8, causal=causal, backend="sdpa")
    sdpa.load_state_dict(naive.state_dict())
    x = torch.randn(2, 20, 64)
    torch.testing.assert_close(naive(x), sdpa(x), atol=1e-5, rtol=1e-4)


def test_manual_mha_matches_torch_multihead_attention():
    """Our MHA against nn.MultiheadAttention, a fully independent implementation."""
    torch.manual_seed(0)
    ours = MultiHeadAttention(64, 8)
    ref = torch.nn.MultiheadAttention(64, 8, bias=False, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight]))
        ref.out_proj.weight.copy_(ours.o_proj.weight)
    x = torch.randn(2, 20, 64)
    torch.testing.assert_close(ours(x), ref(x, x, x, need_weights=False)[0], atol=1e-5, rtol=1e-4)


def test_fused_qkv_equals_separate():
    torch.manual_seed(0)
    mha = MultiHeadAttention(64, 8, causal=True)
    fused = FusedQKVAttention.from_separate(mha)
    x = torch.randn(2, 20, 64)
    torch.testing.assert_close(fused(x), mha(x), atol=1e-6, rtol=1e-5)


def test_gqa_equals_mha_when_kv_heads_equal_q_heads():
    torch.manual_seed(0)
    mha = MultiHeadAttention(64, 8, causal=True)
    gqa = GroupedQueryAttention(AttentionConfig("gqa", 64, 8, num_kv_heads=8, positional_encoding="none"))
    copy_projections(mha, gqa)
    x = torch.randn(2, 20, 64)
    torch.testing.assert_close(gqa(x), mha(x), atol=1e-5, rtol=1e-4)


def test_gqa_equals_mqa_when_one_kv_head():
    torch.manual_seed(0)
    mqa = MultiQueryAttention(64, 8, causal=True)
    gqa = GroupedQueryAttention(AttentionConfig("gqa", 64, 8, num_kv_heads=1, positional_encoding="none"))
    copy_projections(mqa, gqa)
    x = torch.randn(2, 20, 64)
    torch.testing.assert_close(gqa(x), mqa(x), atol=1e-5, rtol=1e-4)


def test_gqa_equals_mha_with_duplicated_kv_weights():
    """GQA is MHA whose KV heads are tied in groups: tile the KV weights and the outputs agree."""
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(AttentionConfig("gqa", 64, 8, num_kv_heads=2, positional_encoding="none"))
    mha = GroupedQueryAttention(AttentionConfig("mha", 64, 8, positional_encoding="none"))
    with torch.no_grad():
        mha.q_proj.weight.copy_(gqa.q_proj.weight)
        mha.o_proj.weight.copy_(gqa.o_proj.weight)
        for name in ["k_proj", "v_proj"]:
            w = getattr(gqa, name).weight.view(2, 8, 64)  # (Hkv, dh, D)
            getattr(mha, name).weight.copy_(w.repeat_interleave(4, dim=0).reshape(64, 64))
    x = torch.randn(2, 20, 64)
    torch.testing.assert_close(gqa(x), mha(x), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("hkv,expected", [(8, 4 * 64 * 64), (2, 2 * 64 * 64 + 2 * 64 * 16), (1, 2 * 64 * 64 + 2 * 64 * 8)])
def test_parameter_counts(hkv, expected):
    m = GroupedQueryAttention(AttentionConfig("gqa", 64, 8, num_kv_heads=hkv))
    assert sum(p.numel() for p in m.parameters()) == expected


def test_sdpa_enable_gqa_matches_repeat():
    q, k, v = torch.randn(1, 8, 10, 16), torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16)
    a = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    b = F.scaled_dot_product_attention(q, repeat_kv(k, 4), repeat_kv(v, 4))
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
