import pytest
import torch

from scratchnet.autograd_tools import format_graph, walk_graph
from scratchnet.layers import Linear


def test_leaf_and_non_leaf():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    assert x.is_leaf and x.grad_fn is None
    assert not y.is_leaf and type(y.grad_fn).__name__ == "SumBackward0"
    y.backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())


def test_graph_walk_reaches_every_leaf():
    layer = Linear(3, 2)
    x = torch.randn(4, 3)
    loss = layer(x).pow(2).mean()
    nodes = walk_graph(loss)
    leaves = sorted(n.leaf_shape for n in nodes if n.leaf_shape is not None)
    assert leaves == [(2,), (3, 2)]
    assert nodes[0].name == "MeanBackward0"
    assert "AccumulateGrad" in format_graph(loss)


def test_gradients_accumulate_until_cleared():
    w = torch.tensor(2.0, requires_grad=True)
    for _ in range(3):
        (3 * w).backward()
    assert w.grad.item() == 9.0  # 3 + 3 + 3: backward adds, it does not overwrite


def test_backward_twice_needs_retain_graph():
    x = torch.tensor(1.5, requires_grad=True)
    y = x.exp()  # exp saves its output for backward
    y.backward(retain_graph=True)
    y.backward()
    with pytest.raises(RuntimeError):
        z = x.exp()
        z.backward()
        z.backward()


def test_detach_and_no_grad_cut_the_graph():
    x = torch.randn(3, requires_grad=True)
    assert x.detach().grad_fn is None and not x.detach().requires_grad
    with torch.no_grad():
        y = x * 2
    assert y.grad_fn is None and not y.requires_grad


def test_linear_gradcheck():
    torch.manual_seed(0)
    layer = Linear(4, 3).double()
    x = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: layer(inp), (x,))
