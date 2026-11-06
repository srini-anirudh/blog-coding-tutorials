import math

import pytest
import torch

from optlab import SGD, Adafactor, Adagrad, Adam, AdamW8bit, Lion, Muon, factored_v, rms


def run_pair(mine, ref, steps=25, shape=(7, 5), seed=0):
    """Drive two optimizers with the same gradients of a fixed quadratic; return both parameters."""
    torch.manual_seed(seed)
    init = torch.randn(shape)
    A = torch.randn(shape)
    a, b = torch.nn.Parameter(init.clone()), torch.nn.Parameter(init.clone())
    oa, ob = mine([a]), ref([b])
    for _ in range(steps):
        for p, o in ((a, oa), (b, ob)):
            p.grad = A * p.detach() + 0.1 * torch.sin(p.detach())
            o.step()
    return a.detach(), b.detach()


@pytest.mark.parametrize("mine,ref", [
    (lambda p: SGD(p, 0.05), lambda p: torch.optim.SGD(p, 0.05)),
    (lambda p: SGD(p, 0.05, momentum=0.9), lambda p: torch.optim.SGD(p, 0.05, momentum=0.9)),
    (lambda p: SGD(p, 0.05, momentum=0.9, nesterov=True), lambda p: torch.optim.SGD(p, 0.05, momentum=0.9, nesterov=True)),
    (lambda p: Adagrad(p, 0.1), lambda p: torch.optim.Adagrad(p, 0.1)),
    (lambda p: Adam(p, 1e-2, weight_decay=0.1), lambda p: torch.optim.AdamW(p, 1e-2, weight_decay=0.1)),
    (lambda p: Adam(p, 1e-2, weight_decay=0.1, decoupled=False), lambda p: torch.optim.Adam(p, 1e-2, weight_decay=0.1)),
])
def test_matches_torch(mine, ref):
    a, b = run_pair(mine, ref)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_rmsprop_matches_torch():
    from optlab import RMSProp

    a, b = run_pair(lambda p: RMSProp(p, 1e-2, beta2=0.99), lambda p: torch.optim.RMSprop(p, 1e-2, alpha=0.99))
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_two_momentum_conventions_differ_by_one_over_one_minus_beta():
    out = {}
    for ema in (False, True):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = SGD([p], 1.0, momentum=0.9, ema=ema)
        for _ in range(300):
            p.grad = torch.ones(1)
            opt.step()
        out[ema] = opt.state[p]["m"].item()
    assert out[False] == pytest.approx(10.0, rel=1e-6)
    assert out[True] == pytest.approx(1.0, rel=1e-6)


def test_bias_correction_first_step():
    for bc, expected in ((True, 1.0), (False, 0.1 / math.sqrt(0.001))):
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        opt = Adam([p], lr=1.0, eps=0.0, bias_correction=bc)
        p.grad = torch.full((1,), 0.37, dtype=torch.float64)
        opt.step()
        assert -p.item() == pytest.approx(expected, rel=1e-9)


def test_lion_moves_every_coordinate_by_exactly_lr():
    p = torch.nn.Parameter(torch.zeros(100))
    opt = Lion([p], lr=1e-3)
    g = torch.randn(100)
    p.grad = g
    opt.step()
    torch.testing.assert_close(p.detach(), -1e-3 * torch.sign(g))
    assert opt.state_bytes() == 4 * 100


def test_factored_v_is_exact_for_rank_one():
    r, c = torch.rand(6) + 0.1, torch.rand(9) + 0.1
    V = torch.outer(r, c)
    torch.testing.assert_close(factored_v(V.mean(1), V.mean(0)), V)


def test_adafactor_state_is_rows_plus_columns():
    p = torch.nn.Parameter(torch.randn(30, 40))
    opt = Adafactor([p])
    p.grad = torch.randn(30, 40)
    opt.step()
    assert opt.state_bytes() == 4 * (30 + 40)


def test_8bit_adamw_tracks_fp32_adamw_and_is_2_bytes():
    # Same gradient sequence for both, so only the state storage differs.
    torch.manual_seed(0)
    grads = [torch.randn(64, 64) * (1 + torch.arange(64.0)) for _ in range(30)]
    a, b = torch.nn.Parameter(torch.zeros(64, 64)), torch.nn.Parameter(torch.zeros(64, 64))
    oa, ob = AdamW8bit([a], 1e-2), Adam([b], 1e-2)
    for g in grads:
        a.grad, b.grad = g, g.clone()
        oa.step(), ob.step()
    assert (a - b).norm() < 0.06 * b.norm()  # re-quantizing every step lets small errors accumulate
    p = torch.nn.Parameter(torch.randn(4096))
    opt = AdamW8bit([p])
    p.grad = torch.randn(4096)
    opt.step()
    assert opt.state_bytes() == 2 * (4096 + 2 * 4)  # 1 byte per value + one fp32 scale per 2048 block, for m and v


def test_muon_step_rms_is_lr_times_scale():
    p = torch.nn.Parameter(torch.zeros(64, 176))
    opt = Muon([p], lr=1.0, momentum=0.0, nesterov=False)
    p.grad = torch.randn(64, 176) @ torch.diag(torch.logspace(0, -3, 176))
    opt.step()
    assert 0.15 < rms(p.detach()).item() < 0.22
    assert opt.state_bytes() == 4 * 64 * 176
