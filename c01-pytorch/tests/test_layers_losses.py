import pytest
import torch
import torch.nn.functional as F
from torch import nn

from scratchnet.layers import MLP, Dropout, Linear, count_parameters
from scratchnet.losses import cross_entropy, log_softmax, log_softmax_naive, mse_loss


def test_linear_matches_nn_linear():
    torch.manual_seed(0)
    ours = Linear(6, 4)
    ref = nn.Linear(6, 4)
    with torch.no_grad():
        ref.weight.copy_(ours.weight.t())  # nn.Linear stores [out, in]
        ref.bias.copy_(ours.bias)
    x = torch.randn(2, 5, 6)
    torch.testing.assert_close(ours(x), ref(x))


def test_mlp_parameters_buffers_and_state_dict():
    model = MLP(2, 16, 3, depth=2)
    names = [n for n, _ in model.named_parameters()]
    assert names == ["blocks.0.0.weight", "blocks.0.0.bias", "blocks.1.0.weight", "blocks.1.0.bias",
                     "head.weight", "head.bias"]
    assert count_parameters(model) == (2 * 16 + 16) + (16 * 16 + 16) + (16 * 3 + 3)
    buffers = [n for n, _ in model.named_buffers()]
    assert buffers == ["norm.mean", "norm.std"]
    assert set(model.state_dict()) == set(names) | set(buffers)
    assert model(torch.randn(7, 2)).shape == (7, 3)


def test_state_dict_round_trip(tmp_path):
    a, b = MLP(2, 8, 3), MLP(2, 8, 3)
    torch.save(a.state_dict(), tmp_path / "m.pt")
    b.load_state_dict(torch.load(tmp_path / "m.pt"))
    x = torch.randn(4, 2)
    torch.testing.assert_close(a(x), b(x))


def test_train_and_eval_change_dropout():
    d = Dropout(0.5)
    x = torch.ones(10_000)
    assert (d(x) == 0).float().mean().item() == pytest.approx(0.5, abs=0.03)
    d.eval()
    assert torch.equal(d(x), x)


def test_mse_matches_and_rejects_silent_broadcast():
    pred, target = torch.randn(8, 3), torch.randn(8, 3)
    torch.testing.assert_close(mse_loss(pred, target), F.mse_loss(pred, target))
    with pytest.raises(ValueError):
        mse_loss(torch.randn(8, 1), torch.randn(8))


def test_log_softmax_matches_and_is_stable():
    z = torch.randn(4, 10) * 5
    torch.testing.assert_close(log_softmax(z), F.log_softmax(z, dim=-1))
    big = torch.tensor([[1000.0, 0.0, -1000.0]])
    assert torch.isfinite(log_softmax(big)).all()
    assert not torch.isfinite(log_softmax_naive(big)).all()


@pytest.mark.parametrize("shape", [(16, 5), (2, 7, 11)])
def test_cross_entropy_matches_torch(shape):
    torch.manual_seed(0)
    logits = torch.randn(*shape, requires_grad=True)
    target = torch.randint(0, shape[-1], shape[:-1])
    target.view(-1)[::3] = -100  # some ignored positions, like padding in a batch
    ours = cross_entropy(logits, target)
    ref = F.cross_entropy(logits.reshape(-1, shape[-1]), target.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(ours, ref)
    (g_ours,) = torch.autograd.grad(ours, logits)
    (g_ref,) = torch.autograd.grad(ref, logits)
    torch.testing.assert_close(g_ours, g_ref)


def test_cross_entropy_gradient_is_softmax_minus_onehot():
    logits = torch.randn(4, 6, requires_grad=True)
    target = torch.tensor([0, 5, 2, 2])
    cross_entropy(logits, target).backward()
    expected = (logits.detach().softmax(-1) - F.one_hot(target, 6)) / 4
    torch.testing.assert_close(logits.grad, expected)
