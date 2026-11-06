"""Position, norms, the KV cache, masks, MoE, linear state, and the small changes."""

import math

import pytest
import torch
import torch.nn.functional as F

from organs.cache import (RollingKVCache, decode_with_window, full_window_reference, interleave, mla_absorption_check,
                          receptive_field)
from organs.memory import read_state, recall_cosine, recall_task, write_delta, write_linear
from organs.moe import BalancedMoE, train_moe
from organs.norms import layer_norm_two_reductions, rms_norm_one_reduction
from organs.position import nope_attention, rope_logit, rope_wavelengths
from organs.small import chunked_cross_entropy, softmax_with_sink


# ---------------------------------------------------------------- position
def test_rope_depends_only_on_offset():
    torch.manual_seed(0)
    q, k = torch.randn(64), torch.randn(64)
    ref = rope_logit(q, k, 7, 3)
    for m in (4, 50, 1000):
        assert rope_logit(q, k, m + 4, m) == pytest.approx(ref, abs=1e-3)
    assert rope_logit(q, k, 3, 7) != pytest.approx(ref, abs=1e-3)


def test_rope_wavelengths():
    w = rope_wavelengths(128, 1e4)
    assert w[0] == pytest.approx(2 * math.pi)
    assert w[-1] == pytest.approx(2 * math.pi * 1e4 ** (126 / 128))


def test_nope_symmetry():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 12, 8, generator=g, dtype=torch.float64)
    w = [torch.randn(8, 8, generator=g, dtype=torch.float64) for _ in range(3)]
    perm = torch.randperm(12, generator=g)
    bi = nope_attention(x, *w, causal=False)
    assert torch.allclose(nope_attention(x[:, perm], *w, causal=False), bi[:, perm])
    ca = nope_attention(x, *w, causal=True)
    assert not torch.allclose(nope_attention(x[:, perm], *w, causal=True), ca[:, perm])


# ---------------------------------------------------------------- norms
def test_layernorm_is_centered_rmsnorm():
    x = torch.randn(16, 32) + 2.0
    assert torch.allclose(layer_norm_two_reductions(x), F.layer_norm(x, (32,)), atol=1e-5)
    assert torch.allclose(rms_norm_one_reduction(x), F.rms_norm(x, (32,), eps=1e-5), atol=1e-6)
    assert torch.allclose(layer_norm_two_reductions(x), rms_norm_one_reduction(x - x.mean(-1, keepdim=True)), atol=1e-5)


# ---------------------------------------------------------------- cache and masks
@pytest.mark.parametrize("window", [1, 5, 40])
def test_rolling_cache_matches_masked_attention(window):
    g = torch.Generator().manual_seed(window)
    q, k, v = (torch.randn(40, 16, generator=g, dtype=torch.float64) for _ in range(3))
    out, nbytes = decode_with_window(q, k, v, window)
    assert torch.allclose(out, full_window_reference(q, k, v, window))
    assert nbytes == 2 * window * 16 * 8


def test_rolling_cache_never_grows():
    c = RollingKVCache(4, 2)
    for t in range(10):
        ks, _ = c.append(torch.full((2,), float(t)), torch.zeros(2))
    assert ks.shape[0] == 4 and sorted(ks[:, 0].tolist()) == [6.0, 7.0, 8.0, 9.0]


def test_receptive_field():
    assert receptive_field(interleave(5, None), 200, 8) == [7, 14, 21, 28, 35]  # L (w - 1)
    assert receptive_field(interleave(4, 3), 200, 8) == [7, 14, 21, 199]
    assert interleave(8, 3) == ["local"] * 3 + ["global"] + ["local"] * 3 + ["global"]


def test_mla_absorption():
    r = mla_absorption_check()
    assert r["absorbed"] == pytest.approx(r["materialized"], rel=1e-12)
    assert abs(r["rope_absorbed_wrong_pos"] - r["rope_materialized"]) > 1.0
    assert r["commutator_norm"] > 0.1


# ---------------------------------------------------------------- MoE
def test_moe_shapes_and_bias_rule():
    torch.manual_seed(0)
    moe = BalancedMoE(8, 16, experts=4, top_k=2, balance="bias", shared=1)
    y, aux = moe(torch.randn(32, 8))
    assert y.shape == (32, 8) and aux.item() == 0
    assert moe.last_load.sum().item() == pytest.approx(1.0)
    load = moe.last_load.clone()
    moe.update_bias(0.1)
    assert torch.equal(moe.bias > 0, load < 0.25) or torch.all(load == 0.25)
    assert not moe.bias.requires_grad  # the bias is never trained by the loss


def test_aux_loss_is_one_when_uniform():
    from tinyllm.moe import load_balancing_loss

    probs = torch.full((8, 4), 0.25)
    idx = torch.tensor([[0], [1], [2], [3]] * 2)
    assert load_balancing_loss(probs, idx, 4).item() == pytest.approx(1.0)


def test_bias_balances_a_head_start():
    none = train_moe("none", 0, steps=150, head_start=4.0)
    bias = train_moe("bias", 0, steps=150, head_start=4.0)
    assert bias["max_over_mean"] < none["max_over_mean"]
    assert bias["max_over_mean"] < 1.6


# ---------------------------------------------------------------- linear state
def test_orthonormal_keys_both_exact():
    keys = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))[0][:10]
    values = torch.randn(10, 16, dtype=torch.float64)
    for s in (write_linear(keys, values), write_delta(keys, values)):
        assert torch.allclose(read_state(s, keys), values)


def test_delta_rule_overwrites():
    keys, old = recall_task(8, 32, 0)
    _, new = recall_task(8, 32, 1)
    s = write_delta(torch.cat([keys, keys]), torch.cat([old, new]))
    lin = write_linear(torch.cat([keys, keys]), torch.cat([old, new]))
    assert recall_cosine(read_state(s, keys), new).mean() > recall_cosine(read_state(lin, keys), new).mean() + 0.1


def test_delta_beats_linear_on_correlated_keys():
    keys, values = recall_task(32, 64, 0, shared=1.0)
    d = recall_cosine(read_state(write_delta(keys, values), keys), values).mean()
    lin = recall_cosine(read_state(write_linear(keys, values), keys), values).mean()
    assert d > lin + 0.1


# ---------------------------------------------------------------- small changes
def test_chunked_cross_entropy_matches():
    h = torch.randn(64, 16, requires_grad=True)
    w = torch.randn(100, 16, requires_grad=True)
    y = torch.randint(0, 100, (64,))
    full = F.cross_entropy(h @ w.T, y)
    gf = torch.autograd.grad(full, (h, w))
    ch = chunked_cross_entropy(h, w, y, 10)
    gc = torch.autograd.grad(ch, (h, w))
    assert ch.item() == pytest.approx(full.item(), rel=1e-6)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(gf, gc))


def test_sink():
    z = torch.randn(3, 5)
    assert torch.allclose(softmax_with_sink(z, float("-inf")), z.softmax(-1))
    assert (softmax_with_sink(z, 2.0).sum(-1) < 1).all()
