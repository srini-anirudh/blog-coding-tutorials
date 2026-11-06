import pytest
import torch
import torch.nn.functional as F

from autogradlab import vjp

dt = torch.float64


def leaf(*s):
    return torch.randn(*s, dtype=dt, requires_grad=True)


def close(ours, theirs):
    for a, b in zip(ours, theirs):
        assert torch.allclose(a, b, atol=1e-12)


def test_linear():
    X, W, b = leaf(5, 4), leaf(4, 3), leaf(3)
    U, s = vjp.linear_forward(X, W, b)
    g = torch.randn_like(U)
    close(vjp.linear_backward(s, g), torch.autograd.grad(X @ W + b, (X, W, b), g))


@pytest.mark.parametrize("name,ref", [("relu", F.relu), ("gelu", F.gelu), ("softmax", lambda z: torch.softmax(z, -1))])
def test_elementwise_and_softmax(name, ref):
    x = leaf(4, 7)
    _, s = getattr(vjp, f"{name}_forward")(x)
    g = torch.randn(4, 7, dtype=dt)
    close([getattr(vjp, f"{name}_backward")(s, g)], torch.autograd.grad(ref(x), (x,), g))


def test_norms_and_projections():
    x, gamma, beta = leaf(4, 9), leaf(9), leaf(9)
    g = torch.randn(4, 9, dtype=dt)
    _, s = vjp.layernorm_forward(x, gamma, beta)
    close(vjp.layernorm_backward(s, g), torch.autograd.grad(F.layer_norm(x, (9,), gamma, beta), (x, gamma, beta), g))
    _, s = vjp.rmsnorm_forward(x, gamma)
    close(vjp.rmsnorm_backward(s, g), torch.autograd.grad(F.rms_norm(x, (9,), gamma, eps=1e-6), (x, gamma), g))
    _, s = vjp.layernorm_forward(x, gamma, beta, eps=0.0)
    gx = vjp.layernorm_backward(s, g)[0]
    assert gx.sum(-1).abs().max() < 1e-12 and (gx * s[0]).sum(-1).abs().max() < 1e-12
    _, s = vjp.rmsnorm_forward(x, gamma, eps=0.0)
    gx = vjp.rmsnorm_backward(s, g)[0]
    assert (gx * s[0]).sum(-1).abs().max() < 1e-12 and gx.sum(-1).abs().max() > 1e-3


def test_cross_entropy_is_p_minus_y():
    z, t = leaf(6, 5), torch.randint(0, 5, (6,))
    _, s = vjp.cross_entropy_forward(z, t)
    close([vjp.cross_entropy_backward(s)], torch.autograd.grad(F.cross_entropy(z, t), (z,)))


def test_attention_and_recompute():
    Q, K, V = leaf(2, 6, 4), leaf(2, 6, 4), leaf(2, 6, 4)
    O, s = vjp.attention_forward(Q, K, V)
    g = torch.randn_like(O)
    ref = torch.autograd.grad(F.scaled_dot_product_attention(Q, K, V), (Q, K, V), g)
    close(vjp.attention_backward(s, g), ref)
    close(torch.autograd.grad(vjp.AttentionRecompute.apply(Q, K, V), (Q, K, V), g), ref)


def count(fn, *inputs):
    with vjp.MatmulCounter() as mc:
        out = fn(*inputs)
        mc.phase = "backward"
        out.sum().backward()
    return mc.counts["forward"], mc.counts["backward"]


def test_matmul_counts():
    W = torch.randn(8, 4, requires_grad=True)
    assert count(lambda x: x @ W, torch.randn(3, 8, requires_grad=True)) == (1, 2)
    assert count(lambda x: x @ W, torch.randn(3, 8)) == (1, 1)
    qkv = [torch.randn(1, 5, 4, requires_grad=True) for _ in range(3)]
    assert count(lambda q, k, v: vjp.attention_forward(q, k, v)[0], *qkv) == (2, 4)
    assert count(vjp.AttentionRecompute.apply, *qkv) == (2, 5)
