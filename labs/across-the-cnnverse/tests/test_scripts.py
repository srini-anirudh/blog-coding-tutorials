from importlib import import_module

import torch

from cnnverse import data


def test_residual_gradient_ratio_stays_near_one():
    g = import_module("05_residual_gradients")
    res = g.grad_ratio(56, **g.VARIANTS["residual, BN"])
    plain = g.grad_ratio(20, **g.VARIANTS["plain, no BN, default init"])
    assert 0.2 < res < 5 and plain < 1e-4


def test_degradation_trainer_runs_and_learns():
    t = import_module("06_degradation")
    (xtr, ytr), (xte, yte) = data.synthetic(256), data.synthetic(128, seed=1)
    r = t.train(8, True, xtr, ytr, xte, yte, epochs=3, bs=64, classes=4, log=lambda *a: None)
    assert r["history"][-1]["train_loss"] < r["history"][0]["train_loss"]
    assert 0 <= r["test_err"] <= 1


def test_augment_keeps_shape_and_values():
    x = data.synthetic(16)[0]
    y = data.augment(x)
    assert y.shape == x.shape and y.dtype == torch.uint8
