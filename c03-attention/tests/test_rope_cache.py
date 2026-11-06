"""RoPE and the KV cache: positions, offsets, and cached decoding == full forward."""

import pytest
import torch

from attnzoo.config import AttentionConfig
from attnzoo.gqa import GroupedQueryAttention
from attnzoo.kv_cache import KVCache, StaticKVCache, kv_cache_bytes
from attnzoo.masks import causal_mask, mask_to_bias
from attnzoo.rope import RotaryEmbedding, apply_rotary


def test_rope_preserves_norm():
    rope = RotaryEmbedding(16, 64)
    q = torch.randn(1, 2, 10, 16)
    cos, sin = rope(torch.arange(10)[None])
    q_rot, _ = apply_rotary(q, q, cos, sin)
    torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1))


def test_rope_scores_depend_only_on_relative_position():
    rope = RotaryEmbedding(16, 256)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m, n):
        cq, sq = rope(torch.tensor([[m]]))
        ck, sk = rope(torch.tensor([[n]]))
        qr, _ = apply_rotary(q, q, cq, sq)
        _, kr = apply_rotary(k, k, ck, sk)
        return (qr * kr).sum().item()

    assert score(10, 7) == pytest.approx(score(110, 107), abs=1e-4)
    assert score(10, 7) != pytest.approx(score(10, 2), abs=1e-3)


def make(hkv=2, **kw):
    torch.manual_seed(0)
    return GroupedQueryAttention(AttentionConfig("gqa", 64, 8, num_kv_heads=hkv, max_seq_len=128, **kw)).eval()


@pytest.mark.parametrize("cache_cls", ["dynamic", "static"])
@pytest.mark.parametrize("hkv", [8, 2, 1])
def test_kv_cached_incremental_equals_full_forward(cache_cls, hkv):
    m = make(hkv)
    x = torch.randn(2, 24, 64)
    full = m(x)
    cache = KVCache() if cache_cls == "dynamic" else StaticKVCache(1, 2, hkv, 64, 8)
    steps = [m(x[:, :10], kv_cache=cache)]  # prefill: Nq = Nk = 10
    for t in range(10, 24):  # decode: Nq = 1, Nk = t + 1
        steps.append(m(x[:, t : t + 1], kv_cache=cache))
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-4)
    assert cache.get_seq_length() == 24


def test_cache_with_chunked_prefill_and_window():
    m = make(2, window_size=6)
    x = torch.randn(1, 30, 64)
    full = m(x)
    cache = KVCache()
    chunks = [m(x[:, s : s + 7], kv_cache=cache) for s in range(0, 30, 7)]  # 7, 7, 7, 7, 2
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-5, rtol=1e-4)


def test_is_causal_shortcut_is_wrong_for_decoding():
    """The bug the bottom-right mask avoids: SDPA's is_causal is aligned top-left."""
    import torch.nn.functional as F

    q, k, v = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 5, 8), torch.randn(1, 1, 5, 8)
    wrong = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # sees key 0 only
    right = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(1, 5))
    torch.testing.assert_close(wrong, v[:, :, :1])
    assert not torch.allclose(wrong, right)


def test_causality_future_tokens_do_not_change_the_past():
    m = make(2)
    x = torch.randn(1, 16, 64)
    y = x.clone()
    y[:, 10:] = torch.randn(1, 6, 64)  # rewrite the future
    torch.testing.assert_close(m(x)[:, :10], m(y)[:, :10])
    assert not torch.allclose(m(x)[:, 10:], m(y)[:, 10:])


def test_cache_bytes_match_formula():
    m = make(2)
    cache = KVCache()
    m(torch.randn(3, 40, 64), kv_cache=cache)
    assert cache.nbytes() == kv_cache_bytes(num_layers=1, batch_size=3, seq_len=40, num_kv_heads=2, head_dim=8, bytes_per_elem=4)


def test_left_padding_with_position_ids():
    """Batched decoding with left padding: positions must skip the pads, and pads are masked."""
    m = make(2)
    x = torch.randn(1, 6, 64)
    alone = m(x)
    padded = torch.cat([torch.zeros(1, 2, 64), x], dim=1)
    attn = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]])
    pos = (attn.cumsum(-1) - 1).clamp_min(0)
    out = m(padded, attention_mask=attn, position_ids=pos)
    torch.testing.assert_close(out[:, 2:], alone, atol=1e-5, rtol=1e-4)


def test_matches_hugging_face_llama_attention():
    """Trusted implementation: transformers' LlamaAttention with the same weights."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

    ours = make(2)
    hf_cfg = LlamaConfig(hidden_size=64, num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=128)
    hf_cfg._attn_implementation = "eager"
    hf = LlamaAttention(hf_cfg, layer_idx=0).eval()
    hf.load_state_dict(ours.state_dict())  # same parameter names: q_proj, k_proj, v_proj, o_proj
    x = torch.randn(2, 20, 64)
    pos = torch.arange(20)[None].expand(2, 20)
    cos, sin = LlamaRotaryEmbedding(hf_cfg)(x, pos)
    ref, _ = hf(x, position_embeddings=(cos, sin), attention_mask=mask_to_bias(causal_mask(20))[None, None])
    torch.testing.assert_close(ours(x), ref, atol=1e-5, rtol=1e-4)
