"""Multi-head latent attention: shapes, absorbed == explicit, cache equivalence and size."""

import pytest
import torch

from attnzoo.config import AttentionConfig
from attnzoo.kv_cache import KVCache, kv_cache_bytes, mla_cache_bytes
from attnzoo.mla import MultiHeadLatentAttention


def make(q_lora_rank=None, backend="sdpa"):
    torch.manual_seed(0)
    cfg = AttentionConfig(
        "mla", hidden_size=64, num_heads=4, head_dim=16, kv_lora_rank=24, qk_rope_head_dim=8,
        q_lora_rank=q_lora_rank, v_head_dim=12, max_seq_len=128, backend=backend,
    )
    return MultiHeadLatentAttention(cfg).eval()


@pytest.mark.parametrize("q_lora_rank", [None, 32])
def test_shapes(q_lora_rank):
    m = make(q_lora_rank)
    x = torch.randn(2, 10, 64)
    c_kv, k_pe = m.compress(x)
    assert c_kv.shape == (2, 1, 10, 24) and k_pe.shape == (2, 1, 10, 8)
    k_nope, v = m.reconstruct(c_kv)
    assert k_nope.shape == (2, 4, 10, 16) and v.shape == (2, 4, 10, 12)
    assert m(x).shape == (2, 10, 64)


def test_absorbed_equals_explicit():
    m = make()
    x = torch.randn(2, 15, 64)
    torch.testing.assert_close(m(x, absorb=True), m(x), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("absorb", [False, True])
def test_cached_decoding_equals_full(absorb):
    m = make()
    x = torch.randn(2, 20, 64)
    full = m(x)
    cache = KVCache()
    steps = [m(x[:, :8], kv_cache=cache, absorb=absorb)]
    steps += [m(x[:, t : t + 1], kv_cache=cache, absorb=absorb) for t in range(8, 20)]
    torch.testing.assert_close(torch.cat(steps, 1), full, atol=1e-5, rtol=1e-4)


def test_cache_holds_latents_and_matches_formula():
    m = make()
    cache = KVCache()
    m(torch.randn(3, 40, 64), kv_cache=cache)
    assert cache.nbytes() == mla_cache_bytes(1, 3, 40, kv_lora_rank=24, rope_head_dim=8, bytes_per_elem=4)
    # The same layer as MHA would cache 2 * H * dh = 128 numbers per token, against r + d_r = 32.
    assert kv_cache_bytes(1, 3, 40, 4, 16, 4) / cache.nbytes() == 4.0


def test_backends_agree():
    x = torch.randn(1, 12, 64)
    ref = make(backend="naive")(x)
    torch.testing.assert_close(make(backend="tiled")(x), ref, atol=1e-5, rtol=1e-4)
