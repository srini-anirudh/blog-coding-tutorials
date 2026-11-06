"""Fitting procedures recover known answers; the training and evaluation path runs end to end on CPU."""

import numpy as np
import pytest
import torch

from scalelab.allocation import HOFFMANN, optimal
from scalelab.data import Windows, epoch_order
from scalelab.fit import fit_parametric, isoflop_minimum, power_law, power_law_with_floor
from scalelab.ladder import LADDER
from scalelab.train import batch_windows, evaluate, lr_at, train_run


def test_power_law_exact():
    x = np.logspace(1, 5, 9)
    A, al = power_law(x, 3.0 * x**-0.4)
    assert (A, al) == (pytest.approx(3.0), pytest.approx(0.4))


def test_power_law_with_floor_recovers_floor():
    x = np.logspace(6, 10, 15)
    E, A, al = power_law_with_floor(x, 1.7 + 400 * x**-0.34, floors=2000)
    assert E == pytest.approx(1.7, abs=0.01)
    assert al == pytest.approx(0.34, abs=0.01)


def test_isoflop_vertex():
    n = np.logspace(6, 8, 7)
    loss = 2 + 0.3 * (np.log10(n) - 7.2) ** 2
    n_opt, l_min = isoflop_minimum(n, loss)
    assert np.log10(n_opt) == pytest.approx(7.2)
    assert l_min == pytest.approx(2.0)
    with pytest.raises(ValueError):
        isoflop_minimum(n, -loss)


def test_isoflop_recovers_hoffmann_exponent():
    cs = np.logspace(18, 21, 4)
    mins = []
    for C in cs:
        ns = optimal(HOFFMANN, C)[0] * np.logspace(-0.6, 0.6, 9)
        mins.append(isoflop_minimum(ns, HOFFMANN(ns, C / (6 * ns)))[0])
    a = np.polyfit(np.log(cs), np.log(mins), 1)[0]
    assert a == pytest.approx(HOFFMANN.a, abs=0.01)


def test_parametric_fit_recovers_exponent_ratio():
    rng = np.random.default_rng(0)
    N = 10 ** rng.uniform(7, 10, 40)
    D = 10 ** rng.uniform(9, 12, 40)
    law, _ = fit_parametric(N, D, HOFFMANN(N, D), steps=1500)
    assert law.a == pytest.approx(HOFFMANN.a, abs=0.02)
    assert np.max(np.abs(law(N, D) / HOFFMANN(N, D) - 1)) < 0.01


def test_windows_and_epochs():
    w = Windows(np.arange(1000, dtype=np.uint16), seq=10, device=torch.device("cpu"))
    x, y = w.get(torch.tensor([0, 3]))
    assert x.shape == (2, 10) and torch.equal(y, x + 1) and x[1, 0] == 30
    order = epoch_order(7, 21, seed=0, device=torch.device("cpu"))
    for e in range(3):
        assert sorted(order[7 * e : 7 * (e + 1)].tolist()) == list(range(7))


def test_schedule_rules():
    assert batch_windows(1e6, 256) * 256 == 16_384
    assert batch_windows(1e12, 256) * 256 == 262_144
    assert lr_at(0, 100, 1.0, 5) == pytest.approx(0.2)
    assert lr_at(99, 100, 1.0, 5) == pytest.approx(0.1, abs=1e-3)


def test_train_run_smoke_and_exact_match_ordering():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 64, 40_000).astype(np.uint16)  # vocab ids < 4096, structure-free
    dev = torch.device("cpu")
    tr, va = Windows(tokens[:30_000], 256, dev), Windows(tokens[30_000:], 256, dev)
    r = train_run(LADDER[0], 16_384, tr, va)
    assert r["tokens"] == 16_384 and r["steps"] == 1
    assert 0 < r["val_loss"] < 10
    ks = [r[f"exact{k}"] for k in (1, 2, 4, 8, 16)]
    assert all(a >= b for a, b in zip(ks, ks[1:]))
    r2 = train_run(LADDER[0], 16_384, tr, va, unique=2_560)
    assert r2["epochs"] == pytest.approx(6.4)


def test_evaluate_perfect_model():
    class Oracle(torch.nn.Module):  # predicts the next token of an arange stream exactly
        def forward(self, x):
            logits = torch.full((*x.shape, 600), -1e4)
            logits.scatter_(-1, (x + 1).unsqueeze(-1), 0.0)
            return type("O", (), {"logits": logits})()

    va = Windows(np.arange(520, dtype=np.uint16), 256, torch.device("cpu"))
    out = evaluate(Oracle(), va)
    assert out["exact16"] == 1.0 and out["val_loss"] < 1e-3
