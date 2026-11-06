import numpy as np
import pytest
import torch

from vlmscale import data
from vlmscale.laws import (chinchilla_loss, compute_optimal, compute_optimal_numeric, crossover_size,
                           decayed_error, fit_chinchilla, inference_frontier, interaction)
from vlmscale.model import TwoTowerConfig
from vlmscale.quality import filter_top, make_pool
from vlmscale.train import train


def test_compute_optimal_closed_form():
    n, d = compute_optimal(1e22, 406.4, 410.7, 0.34, 0.28)
    n2, _ = compute_optimal_numeric(1e22, 1.69, 406.4, 410.7, 0.34, 0.28)
    assert n == pytest.approx(n2, rel=0.01)
    assert 6 * n * d == pytest.approx(1e22)


def test_fit_recovers_exponents():
    ns, ds = np.meshgrid(np.logspace(6, 9, 5), np.logspace(8, 11, 5))
    ns, ds = ns.ravel(), ds.ravel()
    fit = fit_chinchilla(ns, ds, chinchilla_loss(ns, ds, 1.8, 300.0, 900.0, 0.32, 0.30), inits=16)
    assert fit["alpha"] == pytest.approx(0.32, abs=0.02)
    assert fit["beta"] == pytest.approx(0.30, abs=0.02)


def test_interaction_crossover():
    kw = dict(c_benefit=0.05, a_coef=3.0, alpha=0.25, b_coef=2.0, beta=0.2)
    n0 = crossover_size(45e9, **kw)
    assert interaction(n0 * 0.5, 45e9, **kw) > 0 > interaction(n0 * 2, 45e9, **kw)


def test_inference_optimum_moves_with_beta():
    vs = [1, 4, 16, 64, 256]
    best = lambda beta, b: min(inference_frontier(300, 50, vs, 1, b, 0.077, beta), key=lambda x: x["error"])["V"]
    assert best(0.25, 3.0) > best(0.015, 1.0)


def test_decay_repeats_help_less():
    once = decayed_error([1e6, 2e6], 1e6, 1.0, 0.2, 3.0)
    fresh = decayed_error([1e6, 2e6], 2e6, 1.0, 0.2, 3.0)
    assert fresh[1] < once[1] < once[0]


def test_glyph_batch_contract():
    b = data.make_batch(16, torch.Generator().manual_seed(0))
    assert b["pixels"].shape == (16, 3, 32, 32) and b["text"].shape == (16, data.TEXT_LEN)
    assert (b["targets"][:, 1] == b["text"][:, 2]).all()
    assert (b["targets"][:, [0, 2]] == -100).all()
    assert b["text"].max() < data.VOCAB


def test_glyphs_distinct():
    flat = data.BANK.flatten(1)
    dist = (flat[:, None] != flat[None]).sum(-1)
    assert (dist + torch.eye(len(flat), dtype=torch.long) * 99).min() >= 5


def test_quality_filter_cleaner():
    p = make_pool(5000, seed=0)
    x, y = filter_top(p, 0.1)
    kept = p["clean"][p["q"].argsort(descending=True)[:len(x)]]
    assert (y != kept).float().mean() < (p["y"] != p["clean"]).float().mean()


def test_training_learns_coarse_question():
    out = train(TwoTowerConfig(pool_tokens=1, l_dim=32, vocab=data.VOCAB, max_text=3), steps=120, eval_every=60,
                eval_size=256)
    assert out["final"]["color_acc"] > 0.5
