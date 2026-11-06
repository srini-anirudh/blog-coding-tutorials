"""Every CPU backend must agree with the naive reference."""

import pytest
import torch

from attnzoo.attention_naive import naive_attention
from attnzoo.attention_tiled import tiled_attention
from attnzoo.backends import attend
from attnzoo.biases import alibi_bias, alibi_slopes
from attnzoo.masks import padding_mask

CPU_BACKENDS = ["sdpa", "sdpa_math", "sdpa_flash", "tiled"]


def qkv(b=2, hq=8, hkv=2, nq=37, nk=37, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(b, hq, nq, d, generator=g),
        torch.randn(b, hkv, nk, d, generator=g),
        torch.randn(b, hkv, nk, d, generator=g),
    )


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("causal,window", [(False, None), (True, None), (True, 8)])
def test_backends_match_naive(backend, causal, window):
    q, k, v = qkv()
    ref = attend(q, k, v, backend="naive", causal=causal, window_size=window)
    out = attend(q, k, v, backend=backend, causal=causal, window_size=window)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_decode_alignment(backend):
    """A few queries at the end of a long key sequence (decoding with a cache)."""
    q, k, v = qkv(nq=3, nk=40)
    ref = attend(q, k, v, backend="naive", causal=True)
    torch.testing.assert_close(attend(q, k, v, backend=backend, causal=True), ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_padding_and_float_bias(backend):
    q, k, v = qkv()
    pad = padding_mask(torch.tensor([[1] * 37, [1] * 30 + [0] * 7]))
    ref = attend(q, k, v, pad, backend="naive", causal=True)
    torch.testing.assert_close(attend(q, k, v, pad, backend=backend, causal=True), ref, atol=1e-5, rtol=1e-4)
    bias = alibi_bias(alibi_slopes(8), 37, 37)
    ref = attend(q, k, v, bias, backend="naive", causal=True)
    torch.testing.assert_close(attend(q, k, v, bias, backend=backend, causal=True), ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("bq,bk", [(16, 16), (7, 13), (64, 64), (1, 37)])
def test_online_softmax_tiled_equals_naive(bq, bk):
    q, k, v = qkv(nq=50, nk=50)
    ref = naive_attention(q, k, v, causal=True)
    torch.testing.assert_close(tiled_attention(q, k, v, causal=True, block_q=bq, block_k=bk), ref, atol=1e-5, rtol=1e-4)


def test_tiled_skips_future_tiles():
    q, k, v = qkv(nq=64, nk=64)
    trace: list[dict] = []
    tiled_attention(q, k, v, causal=True, block_q=16, block_k=16, trace=trace)
    skipped = sum(t["skipped"] for t in trace)
    assert len(trace) == 16 and skipped == 6  # the 6 tiles strictly above the diagonal


def test_softcap_naive_vs_tiled_and_sdpa_refuses():
    q, k, v = qkv()
    ref = naive_attention(q * 5, k, v, causal=True, softcap=2.0)
    torch.testing.assert_close(tiled_attention(q * 5, k, v, causal=True, softcap=2.0), ref, atol=1e-5, rtol=1e-4)
    with pytest.raises(ValueError):
        attend(q, k, v, backend="sdpa", softcap=2.0)


def test_fully_masked_row_returns_zeros_like_sdpa():
    q, k, v = qkv(b=1, nq=4, nk=4)
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[0] = False
    for backend in ["naive", "sdpa", "tiled"]:
        out = attend(q, k, v, mask, backend=backend)
        assert torch.isfinite(out).all() and out[:, :, 0].abs().max() == 0


def test_dropout_rescales_in_expectation():
    torch.manual_seed(0)
    q, k, v = qkv(b=1, hq=1, hkv=1, nq=8, nk=8)
    v = torch.ones_like(v)  # output = row sum of weights
    outs = torch.stack([naive_attention(q, k, v, dropout_p=0.3) for _ in range(2000)])
    assert abs(outs.mean().item() - 1.0) < 0.02


def test_flash_refuses_cpu():
    q = torch.randn(1, 1, 4, 8)
    with pytest.raises(RuntimeError):
        attend(q, q, q, backend="flash")


def test_uneven_heads_rejected():
    q, k, v = qkv(hq=6, hkv=4)
    with pytest.raises(ValueError):
        attend(q, k, v)


@pytest.mark.parametrize("window", [1, 8, 37, 100])
@pytest.mark.parametrize("nq", [37, 3])
def test_chunked_window_equals_masked_window(window, nq):
    q, k, v = qkv(nq=nq, nk=37)
    ref = attend(q, k, v, backend="naive", causal=True, window_size=window)
    out = attend(q, k, v, backend="window_chunked", causal=True, window_size=window)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_chunked_window_needs_a_window():
    q, k, v = qkv()
    with pytest.raises(ValueError):
        attend(q, k, v, backend="window_chunked", causal=True)
