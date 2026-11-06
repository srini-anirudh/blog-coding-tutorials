"""Linear attention: the parallel form, the recurrence and the random-feature approximation."""

import torch

from attnzoo.config import AttentionConfig
from attnzoo.linear import (
    LinearAttention,
    LinearStateCache,
    causal_linear_attention_parallel,
    causal_linear_attention_recurrent,
    elu_feature_map,
    linear_attention,
    performer_features,
)


def feats(n=12, d=8):
    torch.manual_seed(0)
    return elu_feature_map(torch.randn(2, 3, n, d)), elu_feature_map(torch.randn(2, 3, n, d)), torch.randn(2, 3, n, 5)


def test_reordering_matmuls_gives_the_same_answer():
    q, k, v = feats()
    a = q @ k.transpose(-2, -1)  # (N, N), the quadratic way
    quadratic = (a @ v) / a.sum(-1, keepdim=True)
    torch.testing.assert_close(linear_attention(q, k, v), quadratic, atol=1e-5, rtol=1e-4)


def test_recurrent_equals_parallel():
    q, k, v = feats()
    rec, (s, z) = causal_linear_attention_recurrent(q, k, v)
    torch.testing.assert_close(rec, causal_linear_attention_parallel(q, k, v), atol=1e-5, rtol=1e-4)
    assert s.shape == (2, 3, 8, 5) and z.shape == (2, 3, 8)  # state size does not depend on N


def test_layer_decoding_with_state_equals_parallel():
    torch.manual_seed(0)
    m = LinearAttention(AttentionConfig("linear", 32, 4)).eval()
    x = torch.randn(1, 10, 32)
    full = m(x)
    cache = LinearStateCache()
    steps = [m(x[:, :4], kv_cache=cache)] + [m(x[:, t : t + 1], kv_cache=cache) for t in range(4, 10)]
    torch.testing.assert_close(torch.cat(steps, 1), full, atol=1e-5, rtol=1e-4)
    size_after_10 = cache.nbytes()
    m(torch.randn(1, 50, 32), kv_cache=cache)
    assert cache.nbytes() == size_after_10  # constant memory


def test_performer_error_shrinks_with_more_features():
    torch.manual_seed(0)
    d = 16
    q, k, v = torch.randn(1, 1, 64, d) * 0.5, torch.randn(1, 1, 64, d) * 0.5, torch.randn(1, 1, 64, d)
    exact = torch.softmax(q @ k.transpose(-2, -1) / d**0.5, -1) @ v

    def err(m):
        errs = []
        for seed in range(5):
            w = torch.randn(m, d, generator=torch.Generator().manual_seed(seed))
            approx = linear_attention(performer_features(q, w), performer_features(k, w), v)
            errs.append((approx - exact).abs().mean().item())
        return sum(errs) / len(errs)

    assert err(1024) < err(16)


def test_causality():
    torch.manual_seed(0)
    m = LinearAttention(AttentionConfig("linear", 32, 4)).eval()
    x = torch.randn(1, 10, 32)
    y = x.clone()
    y[:, 6:] = 0
    torch.testing.assert_close(m(x)[:, :6], m(y)[:, :6])
