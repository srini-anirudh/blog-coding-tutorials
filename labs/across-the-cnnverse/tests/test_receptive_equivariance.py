import torch
import torch.nn as nn

from cnnverse import receptive as rf


def test_recursion_matches_gradient_footprint():
    for name, layers in rf.STRATEGIES.items():
        j, r = rf.rf_recursion(layers)[-1]
        size = r + 8 * j + 1
        size += size % 2 == 0
        assert rf.support_width(rf.gradient_map(rf.conv_stack(layers), size)) == r, name


def test_known_receptive_fields():
    assert rf.rf_recursion([(3, 1, 1)] * 8)[-1] == (1, 17)
    alexnet = [(11, 4, 1), (3, 2, 1), (5, 1, 1), (3, 2, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 2, 1)]
    assert rf.rf_recursion(alexnet)[-1] == (32, 195)


def test_effective_rf_is_concentrated():
    layers = [(3, 1, 1)] * 6
    gmap = rf.gradient_map(rf.conv_stack(layers, 4, act=True), 25, 4, weights="random", seed=0)
    assert rf.mass_within(gmap, 3) > (7 / 13) ** 2   # more mass in the centre than its share of area


def test_equivariance_stride1_and_stride2():
    from importlib import import_module
    eq = import_module("03_equivariance")
    x = torch.randn(1, 3, 16, 16)
    c1 = nn.Conv2d(3, 4, 3, padding=1, padding_mode="circular")
    c2 = nn.Conv2d(3, 4, 3, stride=2, padding=1, padding_mode="circular")
    with torch.no_grad():
        assert eq.equivariance_error(c1, x, 1, 1) < 1e-5
        assert eq.equivariance_error(c2, x, 2, 1) < 1e-5
        assert eq.equivariance_error(c2, x, 1, 0) > 0.1


def test_conv_is_a_dense_matrix():
    from importlib import import_module
    eq = import_module("03_equivariance")
    w, x = torch.randn(2, 3, 3, 3), torch.randn(1, 3, 6, 6)
    M = eq.conv_as_dense(w, 6, 6)
    ref = torch.nn.functional.conv2d(x, w, padding=1).flatten()
    assert torch.allclose(M @ x.flatten(), ref, atol=1e-4)
