"""The unified library: every architecture x backend, gradients, causality, precision."""

import pytest
import torch

from attnzoo.config import AttentionConfig
from attnzoo.library import build_attention, cache_elems_per_token, make_cache

CAUSAL_TYPES = ["mha", "mqa", "gqa", "mla", "linear"]
BACKENDS = ["naive", "sdpa", "sdpa_math", "sdpa_flash", "tiled"]


def config(t, **kw):
    kv = {"gqa": 2}.get(t)
    return AttentionConfig(t, 64, 8, num_kv_heads=kv, max_seq_len=256, **kw)


@pytest.mark.parametrize("t", CAUSAL_TYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_every_architecture_runs_on_every_backend(t, backend):
    if t == "mla" and backend == "sdpa_flash":
        pytest.skip("flash kernels need equal q, k, v widths; MLA uses dn + dr for q, k and dv for v")
    torch.manual_seed(0)
    ref = build_attention(config(t, backend="naive")).eval()
    m = build_attention(config(t, backend=backend)).eval()
    m.load_state_dict(ref.state_dict())
    x = torch.randn(2, 33, 64)
    torch.testing.assert_close(m(x), ref(x), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("t", CAUSAL_TYPES)
def test_causality(t):
    torch.manual_seed(0)
    m = build_attention(config(t)).eval()
    x = torch.randn(1, 20, 64)
    y = x.clone()
    y[:, 12:] = torch.randn(1, 8, 64)
    torch.testing.assert_close(m(x)[:, :12], m(y)[:, :12])


@pytest.mark.parametrize("t", CAUSAL_TYPES)
def test_cached_decoding_equals_full(t):
    torch.manual_seed(0)
    m = build_attention(config(t)).eval()
    x = torch.randn(2, 18, 64)
    cache = make_cache(m.config)
    steps = [m(x[:, :6], kv_cache=cache)] + [m(x[:, i : i + 1], kv_cache=cache) for i in range(6, 18)]
    torch.testing.assert_close(torch.cat(steps, 1), m(x), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("t", CAUSAL_TYPES + ["cross"])
def test_gradients_flow_to_every_parameter(t):
    torch.manual_seed(0)
    m = build_attention(config(t))
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = m(x, torch.randn(2, 7, 64)) if t == "cross" else m(x)
    out.pow(2).mean().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("t", CAUSAL_TYPES)
def test_fp32_vs_bf16(t):
    """bf16 keeps 8 mantissa bits (~3 significant digits): expect ~1e-2 relative error, not 1e-6."""
    torch.manual_seed(0)
    m = build_attention(config(t)).eval()
    x = torch.randn(2, 32, 64)
    ref = m(x)
    out = m.to(torch.bfloat16)(x.bfloat16()).float()
    rel = (out - ref).norm() / ref.norm()
    assert rel < 3e-2, rel
    assert rel > 1e-4  # it really did run in lower precision


def test_config_validation():
    with pytest.raises(ValueError):
        AttentionConfig("mha", 64, 8, num_kv_heads=2)
    with pytest.raises(ValueError):
        AttentionConfig("gqa", 64, 8, num_kv_heads=3)
    with pytest.raises(ValueError):
        AttentionConfig("gqa", 60, 8)
    assert AttentionConfig("mqa", 64, 8).num_kv_heads == 1


def test_cache_elems_per_token():
    assert cache_elems_per_token(config("mha")) == 2 * 8 * 8
    assert cache_elems_per_token(config("gqa")) == 2 * 2 * 8
    assert cache_elems_per_token(config("mqa")) == 2 * 1 * 8
    assert cache_elems_per_token(config("mla")) == 64 + 16
