import math

import numpy as np
import pytest
import torch

from stablelab.signal import (chi1, correlation_fixed_point, critical_sigma_w, critical_tanh_point, init_std,
                              jacobian_singular_values, log_variance_walk, nonlinearity_gain, plain_stack)


def test_relu_gain_is_two():
    assert nonlinearity_gain("relu", n=400_000) == pytest.approx(2.0, rel=0.02)
    assert nonlinearity_gain("linear", n=400_000) == pytest.approx(1.0, rel=0.02)


def test_init_std_formulas():
    assert init_std("xavier", 100, 300) == pytest.approx(math.sqrt(2 / 400))
    assert init_std("kaiming", 100, 300) == pytest.approx(math.sqrt(2 / 100))


@pytest.mark.parametrize("scheme,expected", [("kaiming", 1.0), ("xavier", 0.5)])
def test_relu_stack_per_layer_gain(scheme, expected):
    fwd = plain_stack(20, 256, scheme, "relu", batch=128)["fwd"]
    assert (fwd[-1] / fwd[0]) ** (1 / 19) == pytest.approx(expected, rel=0.05)


def test_backward_mirrors_forward_for_linear_stack():
    out = plain_stack(10, 128, "naive", "linear", batch=64)
    # naive N(0, 0.01) at width 128: gain 128 * 1e-4 per layer in both directions
    assert out["bwd"][0] == pytest.approx((128e-4) ** 10, rel=0.5)


def test_log_variance_walk_shape_and_spread():
    walk = log_variance_walk(40, 64, n_nets=32)
    assert walk.shape == (32, 41)
    assert walk[:, 40].std() > walk[:, 10].std()


def test_critical_line():
    sw = critical_sigma_w(0.2)
    assert chi1(sw, 0.2) == pytest.approx(1.0, abs=1e-6)
    c_star, slope = correlation_fixed_point(0.8, 0.2)
    assert c_star == pytest.approx(1.0, abs=1e-6)
    assert slope == pytest.approx(chi1(0.8, 0.2), rel=0.01)  # ordered phase: C'(1) = chi_1
    c_star, _ = correlation_fixed_point(2.5, 0.2)
    assert c_star < 0.9


def test_orthogonal_tanh_is_more_isometric_than_gaussian():
    sw, sb = critical_tanh_point(0.05)
    orth = jacobian_singular_values(32, 64, "tanh", "orthogonal", sw, sb)
    gauss = jacobian_singular_values(32, 64, "tanh", "gaussian", sw, sb)
    assert np.std(orth**2) < np.std(gauss**2)


def test_residual_identity_term_keeps_gradient():
    x = torch.randn(8, 16, requires_grad=True)
    h = x
    for _ in range(50):
        h = h + 0.0 * torch.tanh(h @ torch.randn(16, 16))  # dead branches
    h.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
