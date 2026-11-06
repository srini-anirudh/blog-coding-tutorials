import math

import pytest
import torch
from torch import nn

from nonlin.data import two_spirals
from nonlin.mlp import DeepMLP, decades_per_layer, forward_second_moments, grad_norms_per_layer, preserving_gain


def test_affine_stack_collapses(script):
    mod = script("01_linear_collapse")
    torch.manual_seed(0)
    layers = [nn.Linear(5, 5).double() for _ in range(6)]
    x = torch.randn(10, 5, dtype=torch.float64)
    deep = x
    for layer in layers:
        deep = layer(deep)
    w, b = mod.collapse(layers)
    assert torch.allclose(deep, x @ w.T + b, atol=1e-12)


def test_cubic_net_is_a_cubic_polynomial(script):
    """One hidden layer of x^3 - x: the fitted function lies in span{1, x, x^2, x^3} at any width."""
    torch.manual_seed(0)
    model = DeepMLP(1, 50, 1, 1, "cubic", init="xavier").double()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_()
    x = torch.linspace(-1, 1, 101, dtype=torch.float64)
    y = model(x.unsqueeze(1)).squeeze(1).detach()
    mod = script("01_linear_collapse")
    assert mod.best_cubic_mse(x, y) < 1e-18 * max(1.0, y.pow(2).mean().item())


def test_preserving_gain_relu_is_sqrt2():
    for q in (0.25, 4.0):
        assert preserving_gain("relu", q, n=200_000) == pytest.approx(math.sqrt(2), rel=0.01)


def test_gelu_gain_depends_on_scale():
    assert preserving_gain("gelu", 0.25, n=200_000) > preserving_gain("gelu", 4.0, n=200_000) + 0.2


def test_relu_xavier_halves_second_moment():
    torch.manual_seed(0)
    m = forward_second_moments(DeepMLP(128, 128, 2, 10, "relu", init="xavier"), torch.randn(2048, 128))
    ratio = (m[-1] / m[0]) ** (1 / 9)
    assert ratio == pytest.approx(0.5, abs=0.05)
    torch.manual_seed(0)
    m = forward_second_moments(DeepMLP(128, 128, 2, 10, "relu", init="he"), torch.randn(2048, 128))
    assert (m[-1] / m[0]) ** (1 / 9) == pytest.approx(1.0, abs=0.1)


def test_sigmoid_gradient_decays_relu_does_not():
    torch.manual_seed(0)
    x, y = torch.randn(256, 64), torch.randint(10, (256,))
    sig = grad_norms_per_layer(DeepMLP(64, 64, 10, 12, "sigmoid", init="xavier"), x, y)
    rel = grad_norms_per_layer(DeepMLP(64, 64, 10, 12, "relu", init="he"), x, y)
    assert len(sig["signal"]) == len(sig["weight"]) == 12
    assert decades_per_layer(sig["signal"]) < -0.4
    assert abs(decades_per_layer(rel["signal"])) < 0.1


def test_decades_per_layer_slope():
    norms = [10.0**-k for k in range(10)][::-1]  # input side smallest: 1e-9 ... 1
    assert decades_per_layer(norms) == pytest.approx(-1.0)


def test_unknown_init():
    with pytest.raises(ValueError):
        DeepMLP(2, 4, 2, 1, "relu", init="bogus")


def test_spirals_shapes_and_balance():
    x, y = two_spirals(100, seed=1)
    assert x.shape == (100, 2) and y.float().mean().item() == 0.5
