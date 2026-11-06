import math

import pytest
import torch

from autogradlab.chain import chain_cost, gradient_three_ways, jvp_vjp_dot_test, mlp, multiply_chain, random_jacobians
from autogradlab.scalar import Value, backward, forward_mode_gradient, post_example


def test_post_example_matches_post_and_torch():
    g = post_example()
    edges = backward(g["L"])
    assert g["L"].value == pytest.approx((math.tanh(2.0) + 2 - 2) ** 2)
    skip = [c for child, parent, c in edges if child == "r" and parent == "x"][0]
    assert round(skip, 2) == 1.93 and round(g["x"].grad, 2) == 2.13  # the numbers the post quotes
    t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in dict(w=1.5, x=2.0, b=-1.0, y=2.0).items()}
    (((t["w"] * t["x"] + t["b"]).tanh() + t["x"] - t["y"]) ** 2).backward()
    for k in t:
        assert g[k].grad == pytest.approx(t[k].grad.item(), abs=1e-12)


def test_forward_mode_needs_one_sweep_per_input_and_agrees():
    g = post_example()
    backward(g["L"])
    fwd, sweeps = forward_mode_gradient(dict(w=1.5, x=2.0, b=-1.0, y=2.0))
    assert sweeps == 4
    for k, v in fwd.items():
        assert v == pytest.approx(g[k].grad, abs=1e-12)


def test_wrong_visit_order_gives_partial_sum():
    a = Value(0.5)
    s = a.tanh()
    p, q = s * 2.0, s.exp()
    out = p + q
    backward(out)
    right = a.grad
    backward(out, order=[out, p, s, q])
    assert right == pytest.approx((2 + math.exp(math.tanh(0.5))) * (1 - math.tanh(0.5) ** 2))
    assert a.grad == pytest.approx(2 * (1 - math.tanh(0.5) ** 2))


def test_chain_cost_matches_post():
    c = chain_cost([4096, 4096, 4096, 1])
    assert c["reverse"] == 2 * 4096**2  # 33.6 million
    assert c["forward"] == 4096**3 + 4096**2  # 68.7 billion
    square = chain_cost([64, 64, 64, 64])
    assert square["reverse"] == square["forward"]


def test_both_orders_same_product():
    J = random_jacobians([50, 40, 30, 1])
    assert torch.allclose(multiply_chain(J, "reverse"), multiply_chain(J, "forward"))


def test_three_ways_agree():
    model = mlp(8, 16, 2).double()
    r = gradient_three_ways(model, torch.randn(1, 8, dtype=torch.float64))
    assert torch.allclose(r["reverse"]["grad"], r["forward"]["grad"], atol=1e-12)
    assert torch.allclose(r["reverse"]["grad"], r["finite_diff"]["grad"], atol=1e-6)
    assert r["forward"]["sweeps"] == 8 and r["reverse"]["sweeps"] == 1


def test_dot_product_test():
    W = torch.randn(6, 5, dtype=torch.float64)
    a, b = jvp_vjp_dot_test(lambda x: torch.tanh(x @ W), torch.randn(3, 6, dtype=torch.float64))
    assert a == pytest.approx(b, rel=1e-10)
