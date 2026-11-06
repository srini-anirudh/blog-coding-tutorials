import math

import pytest
import torch
import torch.nn.functional as F

from nonlin import activations as A

X = torch.linspace(-10, 10, 4001, dtype=torch.float64)

TORCH = {
    "sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": F.relu, "leaky_relu": F.leaky_relu, "elu": F.elu,
    "selu": F.selu, "relu6": F.relu6, "gelu": F.gelu, "gelu_tanh": lambda t: F.gelu(t, approximate="tanh"),
    "silu": F.silu, "mish": F.mish, "hard_swish": F.hardswish, "softplus": F.softplus,
}


@pytest.mark.parametrize("name", sorted(TORCH))
def test_matches_torch(name):
    assert torch.allclose(A.get(name)(X), TORCH[name](X), atol=1e-12, rtol=0)


def test_quick_gelu_formula():
    assert torch.allclose(A.quick_gelu(X), X * torch.sigmoid(1.702 * X), atol=1e-12)


def test_relu2_is_x_times_relu():
    assert torch.equal(A.relu2(X), X * A.relu(X))


def test_unknown_name():
    with pytest.raises(ValueError):
        A.get("nope")


def test_derivatives():
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float64)
    assert torch.equal(A.derivative(A.relu, x), torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64))
    s = torch.sigmoid(x)
    assert torch.allclose(A.derivative(A.sigmoid, x), s * (1 - s))
    # ReLU at exactly 0: PyTorch-style subgradient 0, as the post says.
    assert A.derivative(A.relu, torch.zeros(1)).item() == 0.0
    # Leaky ReLU keeps a gradient on the negative side.
    assert A.derivative(A.leaky_relu, torch.tensor([-3.0])).item() == pytest.approx(0.01)


def test_sigmoid_derivative_peaks_at_quarter():
    assert A.derivative(A.sigmoid, X).max().item() == pytest.approx(0.25, abs=1e-6)
    assert A.derivative(A.tanh, X).max().item() == pytest.approx(1.0, abs=1e-6)


def test_gelu_dip_and_slope():
    y = A.gelu(X)
    i = int(torch.argmin(y))
    assert X[i].item() == pytest.approx(-0.75, abs=0.01)
    assert y[i].item() == pytest.approx(-0.170, abs=0.001)
    assert A.derivative(A.gelu, X).max().item() == pytest.approx(1.129, abs=0.001)


def test_silu_dip_is_deeper_and_further_left():
    y = A.silu(X)
    i = int(torch.argmin(y))
    assert X[i].item() == pytest.approx(-1.28, abs=0.01)
    assert y[i].item() == pytest.approx(-0.278, abs=0.001)


def test_gaps_between_gelu_variants():
    g = A.gelu(X)
    assert (A.gelu_tanh(X) - g).abs().max().item() < 5e-4
    assert 0.015 < (A.quick_gelu(X) - g).abs().max().item() < 0.025
    assert (A.silu(X) - g).abs().max().item() > 0.15  # SiLU is not within 0.02 of GELU


def test_swish_limits():
    assert torch.allclose(A.swish(X, 1e-6), X / 2, atol=1e-4)
    assert torch.allclose(A.swish(X, 1e4), A.relu(X), atol=1e-3)


def test_stochastic_gate_converges_to_gelu():
    gen = torch.Generator().manual_seed(0)
    x = torch.linspace(-3, 3, 13, dtype=torch.float64)
    err = (A.stochastic_gate_sample(x, 50_000, gen) - A.gelu(x)).abs().max().item()
    assert err < 0.04
    assert err < (A.stochastic_gate_sample(x, 10, gen) - A.gelu(x)).abs().max().item()


def test_softplus_no_overflow():
    big = torch.tensor([1000.0, -1000.0])
    out = A.softplus(big)
    assert torch.isfinite(out).all() and out[0].item() == pytest.approx(1000.0) and out[1].item() == 0.0


def test_module_wrapper():
    m = A.Activation("gelu")
    assert torch.equal(m(X), A.gelu(X)) and "gelu" in repr(m)
    assert math.isclose(A.SELU_LAMBDA, 1.0507, abs_tol=1e-4)
