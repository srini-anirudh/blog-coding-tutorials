import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from autogradlab.checkpoint import backprop_segments, grads_of, make_chain, plain_backprop, segment_bounds
from autogradlab.custom import BuggyLinear, MyLinear, RoundSTE, dot_test
from autogradlab.saved import graph_nodes, saved_bytes
from autogradlab.vjp import AttentionRecompute, attention_forward

ACT = 8 * 16 * 4


def x():
    return torch.randn(8, 16, requires_grad=True)


def test_saved_bytes_per_op():
    W = torch.randn(16, 16, requires_grad=True)
    assert saved_bytes(lambda v: v @ W, x(), exclude=[W])[0] == ACT
    assert saved_bytes(lambda v: v + v.detach().clone().requires_grad_(True), x())[0] == 0
    assert saved_bytes(lambda v: v * v.detach().clone().requires_grad_(True), x())[0] == 2 * ACT
    assert saved_bytes(F.relu, x())[0] == ACT


def test_attention_recompute_saves_no_p():
    qkv = [torch.randn(1, 2, 64, 8, requires_grad=True) for _ in range(3)]
    plain = saved_bytes(lambda q, k, v: attention_forward(q, k, v)[0], *qkv)[0]
    recomp = saved_bytes(AttentionRecompute.apply, *qkv)[0]
    assert recomp == 3 * 64 * 8 * 2 * 4
    assert plain == recomp + 2 * 64 * 64 * 4


def test_detach_and_no_grad_cut_the_graph():
    w = torch.randn(16, 16, requires_grad=True)
    full = (x() @ w).tanh() @ w
    cut = (x() @ w).tanh().detach() @ w
    with torch.no_grad():
        h = (x() @ w).tanh()
    assert graph_nodes(cut) < graph_nodes(full) and graph_nodes(h) == 0


def test_version_counter():
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y = torch.sigmoid(x())
        y.mul_(2)
        y.sum().backward()
    W = torch.randn(16, 16, requires_grad=True)
    (x() @ W).relu_().sum().backward()  # safe: the matmul saved its input, not its output


def test_grad_accumulation_exact_with_layernorm():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 1)).double()
    data, y = torch.randn(12, 4, dtype=torch.float64), torch.randn(12, 1, dtype=torch.float64)
    F.mse_loss(model(data), y).backward()
    full = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for xs, ys in zip(data.split(4), y.split(4)):
        (F.mse_loss(model(xs), ys) / 3).backward()
    for a, b in zip(full, model.parameters()):
        assert torch.allclose(a, b.grad, atol=1e-12)


def test_segment_bounds():
    assert segment_bounds(10, 3) == [(0, 3), (3, 7), (7, 10)]
    assert segment_bounds(4, 4) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_checkpointing_same_gradients_less_memory():
    layers = make_chain(9, 16)
    inp = torch.randn(4, 16)
    base = plain_backprop(layers, inp)
    ref = grads_of(layers).clone()
    for k in [1, 3, 9]:
        r = backprop_segments(layers, inp, k)
        assert torch.equal(grads_of(layers), ref)
        assert r["layer_evals"] == 18 and r["loss"] == base["loss"]
    assert backprop_segments(layers, inp, 3)["peak_bytes"] < 0.5 * base["peak_bytes"]


def test_custom_backward_checks():
    xx = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    assert gradcheck(MyLinear.apply, (xx, w))
    assert not gradcheck(BuggyLinear.apply, (xx, w), raise_exception=False)
    a, b = dot_test(MyLinear.apply, xx.detach(), w.detach())
    assert a == pytest.approx(b, rel=1e-6)
    a, b = dot_test(BuggyLinear.apply, xx.detach(), w.detach())
    assert a != pytest.approx(b, rel=1e-3)


def test_straight_through_passes_gradient():
    t = torch.tensor([0.4, 1.6], requires_grad=True)
    RoundSTE.apply(t).sum().backward()
    assert torch.equal(t.grad, torch.ones(2))
