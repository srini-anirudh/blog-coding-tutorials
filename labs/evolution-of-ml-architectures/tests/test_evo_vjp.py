import math

import pytest
import torch
import torch.nn.functional as F

from archlab import vjp


@pytest.fixture(autouse=True)
def float64():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(torch.float32)


def grads(fn, *xs):
    xs = [x.detach().clone().requires_grad_(True) for x in xs]
    fn(*xs).backward()
    return [x.grad for x in xs]


def close(a, b):
    return torch.allclose(a, b, atol=1e-10)


def test_linear_and_activations():
    X, W, Yb = torch.randn(4, 3), torch.randn(3, 2), torch.randn(4, 2)
    Xb, Wb, _ = vjp.linear(X, W, Yb)
    aX, aW = grads(lambda X, W: (X @ W * Yb).sum(), X, W)
    assert close(Xb, aX) and close(Wb, aW)
    z = torch.randn(20)
    fns = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "gelu": F.gelu, "silu": F.silu}
    for name, f in fns.items():
        assert close(vjp.ACT_DERIVATIVES[name](z), grads(lambda z: f(z).sum(), z)[0]), name


def test_softmax_cross_entropy_is_p_minus_y():
    z, y = torch.randn(5, 7), torch.randint(0, 7, (5,))
    assert close(vjp.softmax_cross_entropy(z, y), grads(lambda z: F.cross_entropy(z, y), z)[0])


def test_norms():
    x, gamma, yb = torch.randn(3, 8), torch.randn(8), torch.randn(3, 8)
    xb = vjp.layernorm(x, gamma, yb)[0]
    assert close(xb, grads(lambda x: (F.layer_norm(x, (8,), gamma) * yb).sum(), x)[0])
    rms = lambda x: x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * gamma  # noqa: E731
    assert close(vjp.rmsnorm(x, gamma, yb), grads(lambda x: (rms(x) * yb).sum(), x)[0])


def test_conv_backward_is_transposed_conv():
    X, K, Yb = torch.randn(2, 3, 8, 8), torch.randn(4, 3, 3, 3), torch.randn(2, 4, 6, 6)
    Xb, Kb = vjp.conv2d(X, K, Yb)
    aX, aK = grads(lambda X, K: (F.conv2d(X, K) * Yb).sum(), X, K)
    assert close(Xb, aX) and close(Kb, aK) and close(Xb, F.conv_transpose2d(Yb, K))


def test_attention_and_flash_identity():
    Q, K, V, Ob = (torch.randn(6, 4) for _ in range(4))
    mQ, mK, mV, D1, D2 = vjp.attention(Q, K, V, Ob)
    aQ, aK, aV = grads(lambda Q, K, V: (torch.softmax(Q @ K.T / 2, -1) @ V * Ob).sum(), Q, K, V)
    assert close(mQ, aQ) and close(mK, aK) and close(mV, aV) and close(D1, D2)


def test_bptt_and_ssm_backward_scan():
    T, d = 6, 4
    xs, h0, Wh, Wx, b, hbs = torch.randn(T, d), torch.randn(d), torch.randn(d, d), torch.randn(d, d), torch.randn(d), torch.randn(T, d)

    def loss(Wh, Wx, h0):
        h, out = h0, 0
        for t in range(T):
            h = torch.tanh(Wh @ h + Wx @ xs[t] + b)
            out = out + (h * hbs[t]).sum()
        return out

    for m, a in zip(vjp.rnn_bptt(xs, h0, Wh, Wx, b, hbs), grads(loss, Wh, Wx, h0)):
        assert close(m, a)
    a, bb, c, yb = torch.rand(9, 3), torch.randn(9, 3), torch.randn(9, 3), torch.randn(9, 3)

    def scan_loss(a, bb, c):
        h, out = torch.zeros(3), 0
        for t in range(9):
            h = a[t] * h + bb[t]
            out = out + (c[t] * h * yb[t]).sum()
        return out

    for m, g in zip(vjp.ssm_scan_backward(a, bb, c, yb), grads(scan_loss, a, bb, c)):
        assert close(m, g)


def test_scale_keeps_softmax_alive():
    q, k = torch.randn(500, 128), torch.randn(32, 128)
    trace = lambda s: (torch.softmax(s, -1) * (1 - torch.softmax(s, -1))).sum(-1).mean()  # noqa: E731
    assert trace(q @ k.T / math.sqrt(128)) > 3 * trace(q @ k.T)
