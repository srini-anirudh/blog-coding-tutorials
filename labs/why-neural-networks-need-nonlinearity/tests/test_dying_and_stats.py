import json

import pytest
import torch
from torch import nn

from nonlin.data import two_spirals
from nonlin.mlp import DeepMLP
from nonlin.stats import Results, exact_zero_fraction, int8_per_tensor_error, kurtosis, max_over_rms


def _kill_unit(act: str):
    torch.manual_seed(0)
    model = DeepMLP(2, 8, 2, 1, act, init="he")
    with torch.no_grad():
        model.hidden[0].bias[3] = -100.0  # pre-activation negative for every input
    x, y = two_spirals(200)
    nn.functional.cross_entropy(model(x), y).backward()
    layer = model.hidden[0]
    return layer.weight.grad[3], layer.bias.grad[3]


def test_dead_relu_gets_exactly_zero_gradient():
    gw, gb = _kill_unit("relu")
    assert torch.count_nonzero(gw) == 0 and gb.item() == 0.0


def test_leaky_unit_still_gets_a_gradient():
    gw, gb = _kill_unit("leaky_relu_0.1")
    assert torch.count_nonzero(gw) > 0 and gb.item() != 0.0


def test_drift_run_relu_weights_frozen(script):
    mod = script("03_dying_relu")
    x, y = two_spirals(400)
    r = mod.drift_run("relu", 40, 40, 0.05, 0.1, 0, x, y)
    assert r["inactive_at_mid"] > 0
    assert r["recovered"] == 0 and r["max_weight_change_of_mid_inactive"] == 0.0


def test_stats():
    torch.manual_seed(0)
    g = torch.randn(200_000)
    assert abs(kurtosis(g)) < 0.05
    assert kurtosis(torch.distributions.StudentT(5.0).sample((200_000,))) > 2
    assert exact_zero_fraction(torch.relu(g)) == pytest.approx(0.5, abs=0.01)
    assert int8_per_tensor_error(g) < 0.02
    spiky = g.clone()
    spiky[0] = 1000.0
    assert int8_per_tensor_error(spiky) > 5 * int8_per_tensor_error(g)
    assert max_over_rms(spiky) > 100


def test_results_writer(tmp_path):
    res = Results("demo", tmp_path)
    res.print("hello", 1)
    res.data["x"] = 2
    res.save()
    txt = (tmp_path / "demo.txt").read_text().splitlines()
    assert txt[0].startswith("# hardware:") and txt[1].startswith("# command:") and txt[2] == "hello 1"
    assert json.loads((tmp_path / "demo.json").read_text())["x"] == 2
