import pytest
import torch
import torch.nn.functional as F

from archlab.gradients import deep_stack_input_grad, lstm_cell_path, rnn_jacobian_norms
from archlab.seqmodels import SeqModel, SSMMixer, key_recall_batch
from archlab.vision import CNN, MLP, TinyViT, receptive_field_width, shapes_batch


def test_linear_rnn_gradient_is_lambda_power():
    n = rnn_jacobian_norms(0.9, 50, nonlinear=False)
    assert abs(n[49].item() - 0.9 ** 50) < 1e-5


def test_residual_and_forget_gate_keep_gradients():
    assert deep_stack_input_grad(20, 32, "sigmoid", residual=False) < 1e-8
    assert deep_stack_input_grad(20, 32, "sigmoid", residual=True) > 0.5
    assert lstm_cell_path(3.0, 50)[49] > lstm_cell_path(0.0, 50)[49] * 1e6


def test_receptive_field():
    assert [receptive_field_width(L) for L in range(1, 5)] == [3, 5, 7, 9]


def test_parallel_and_sequential_ssm_agree():
    torch.manual_seed(0)
    m = SSMMixer(16, scan="sequential")
    x = torch.randn(2, 21, 16)
    y1 = m(x)
    m.scan = "parallel"
    assert torch.allclose(y1, m(x), atol=1e-5)


@pytest.mark.parametrize("kind", ["rnn", "lstm", "attention", "lti_ssm", "selective_ssm"])
def test_seq_models_shapes(kind):
    x, y = key_recall_batch(3, 5)
    assert x.shape == (3, 8) and (x[:, 0] == 16).all() and (x[:, 1] == y).all()
    assert SeqModel(kind, 18, 16, d=16, max_len=8)(x).shape == (3, 16)


def test_attention_learns_short_recall():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = SeqModel("attention", 18, 16, d=32, max_len=7)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(200):
        x, y = key_recall_batch(32, 4, generator=gen)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = key_recall_batch(256, 4, generator=gen)
    assert (model(x).argmax(-1) == y).float().mean() > 0.9


def test_vision_models_and_equivariance():
    x, y = shapes_batch(4, (0, 5))
    for cls in (MLP, CNN, TinyViT):
        assert cls()(x).shape == (4, 4)
    assert MLP().body[1].weight.numel() == 32_768
    cnn = CNN(padding_mode="circular")
    with torch.no_grad():
        assert torch.allclose(cnn.feature_map(torch.roll(x, 3, 3)), torch.roll(cnn.feature_map(x), 3, 3), atol=1e-5)
