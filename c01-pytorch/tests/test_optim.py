import copy

import pytest
import torch

from scratchnet import optim as manual
from scratchnet.layers import MLP
from scratchnet.losses import cross_entropy


def _run(model, opt, steps=25):
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        x = torch.randn(32, 2, generator=g)
        y = torch.randint(0, 3, (32,), generator=g)
        opt.zero_grad()
        cross_entropy(model(x), y).backward()
        opt.step()
    return model


@pytest.mark.parametrize(
    "ours,ref",
    [
        (lambda p: manual.SGD(p, lr=0.1), lambda p: torch.optim.SGD(p, lr=0.1)),
        (lambda p: manual.SGD(p, lr=0.05, momentum=0.9, weight_decay=1e-3),
         lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=1e-3)),
        (lambda p: manual.AdamW(p, lr=1e-2, weight_decay=0.1), lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=0.1)),
    ],
    ids=["sgd", "sgd-momentum-wd", "adamw"],
)
def test_manual_optimizers_match_torch(ours, ref):
    torch.manual_seed(0)
    a = MLP(2, 16, 3)
    b = copy.deepcopy(a)
    _run(a, ours(list(a.parameters())))
    _run(b, ref(b.parameters()))
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-6)


def test_zero_grad_sets_none():
    model = MLP(2, 4, 3)
    opt = manual.SGD(model.parameters(), lr=0.1)
    cross_entropy(model(torch.randn(3, 2)), torch.tensor([0, 1, 2])).backward()
    assert all(p.grad is not None for p in model.parameters())
    opt.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_adamw_first_step_is_lr_sized():
    # With bias correction, step 1 moves every weight by about lr * sign(g).
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    p.grad = torch.tensor([1e-4, -5.0, 300.0])
    manual.AdamW([p], lr=0.01, weight_decay=0.0).step()
    torch.testing.assert_close(p.detach(), torch.tensor([0.99, -1.99, 2.99]), rtol=0, atol=1e-5)


def test_clip_grad_norm_matches_torch():
    torch.manual_seed(0)
    a = MLP(2, 8, 3)
    b = copy.deepcopy(a)
    x, y = torch.randn(16, 2) * 50, torch.randint(0, 3, (16,))
    for m in (a, b):
        cross_entropy(m(x), y).backward()
    n_ours = manual.clip_grad_norm(a.parameters(), 1.0)
    n_ref = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0)
    torch.testing.assert_close(n_ours, n_ref)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad)
    assert manual.grad_norm(a.parameters()).item() == pytest.approx(1.0, rel=1e-4)


def test_manual_optimizer_state_dict_round_trip():
    torch.manual_seed(0)
    a = MLP(2, 8, 3)
    opt_a = manual.AdamW(a.parameters(), lr=1e-2)
    _run(a, opt_a, steps=3)
    b = copy.deepcopy(a)
    opt_b = manual.AdamW(b.parameters(), lr=1e-2)
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))
    _run(a, opt_a, steps=3)
    _run(b, opt_b, steps=3)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb)
