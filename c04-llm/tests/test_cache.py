"""The KV cache must change speed, never results."""

import pytest
import torch

from c04_helpers import random_ids, small_model
from tinyllm import KVCache


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_token_by_token_with_cache_equals_full_forward(backend):
    model = small_model(attn_backend=backend)
    ids = random_ids(2, 15)
    full = model(ids).logits  # (2, 15, V)

    cache = KVCache()
    steps = [model(ids[:, t : t + 1], past_key_values=cache).logits for t in range(ids.shape[1])]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-5)
    assert cache.get_seq_length() == 15


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_chunked_prefill_equals_full_forward(backend):
    """Prompt in chunks of 4 on top of a cache: needs the shifted causal mask."""
    model = small_model(attn_backend=backend)
    ids = random_ids(1, 14)
    full = model(ids).logits
    cache = KVCache()
    chunks = [model(ids[:, s : s + 4], past_key_values=cache).logits for s in range(0, 14, 4)]
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-5, rtol=1e-5)


def test_cache_stores_kv_heads_not_query_heads():
    model = small_model(num_heads=4, num_kv_heads=1)
    out = model(random_ids(2, 9), use_cache=True)
    k = out.past_key_values.keys[0]
    assert k.shape == (2, 1, 9, model.config.head_dim)  # (B, H_kv, T, d)


def test_cached_generation_matches_uncached_greedy():
    model = small_model()
    prompt = random_ids(2, 5)
    a = model.generate(prompt, max_new_tokens=20, use_cache=False)
    b = model.generate(prompt, max_new_tokens=20, use_cache=True)
    assert torch.equal(a, b)


def test_cached_generation_matches_uncached_sampling():
    model = small_model()
    prompt = random_ids(1, 5)
    kw = dict(max_new_tokens=20, temperature=0.9, top_k=20, top_p=0.95, seed=123)
    assert torch.equal(model.generate(prompt, use_cache=False, **kw), model.generate(prompt, use_cache=True, **kw))


def test_eos_stops_generation():
    model = small_model()
    prompt = random_ids(1, 3)
    first = model.generate(prompt, max_new_tokens=1)[0, -1].item()
    out = model.generate(prompt, max_new_tokens=10, eos_token_id=first)
    assert out.shape[1] == 4  # stopped right after emitting EOS
