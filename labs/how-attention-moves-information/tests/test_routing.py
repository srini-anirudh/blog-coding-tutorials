"""Projections, scaling, masks and reach: the mechanisms of the first half of the post."""

import math

import pytest
import torch

from attnmoves import reach
from attnmoves.projections import identifiability_gap, permutation_check, raw_self_attention_stats, train_router
from attnmoves.scaling import jacobian_norm_direct, shift_invariance, softmax_stats


def test_raw_scores_are_symmetric_and_self_focused():
    s = raw_self_attention_stats()
    assert s["max |S - S^T|"] == 0.0
    assert s["rows whose argmax is self"] == 1.0


def test_only_the_qk_product_matters():
    r = identifiability_gap()
    assert r["max score change after reparameterising"] < 1e-9
    assert r["max weight change in W_Q"] > 1.0
    assert r["rank of W_Q W_K^T"] == r["d_k"]


def test_permutation_equivariance_until_positions():
    r = permutation_check()
    assert r["no positions: max |f(Px) - P f(x)|"] < 1e-5
    assert r["positions added: max |f(Px + p) - P f(x + p)|"] > 0.1


def test_separate_projections_solve_a_directed_cycle_and_tied_cannot():
    assert train_router("separate", "cycle3", steps=400)["accuracy"] > 0.95
    assert train_router("tied", "cycle3", steps=400)["accuracy"] < 2 / 3 + 0.05  # the symmetric cap


@pytest.mark.parametrize("d_k", [16, 128, 1024])
def test_logit_std_is_sqrt_dk_and_scaling_removes_it(d_k):
    assert softmax_stats(d_k, scaled=False)["logit std"] == pytest.approx(math.sqrt(d_k), rel=0.05)
    assert softmax_stats(d_k, scaled=True)["logit std"] == pytest.approx(1.0, rel=0.05)


def test_unscaled_softmax_collapses_with_dk():
    e = [softmax_stats(d, scaled=False)["entropy (nats)"] for d in (16, 128, 1024)]
    assert e[0] > e[1] > e[2]
    assert softmax_stats(1024, scaled=True)["entropy (nats)"] > 4.0


def test_jacobian_closed_form():
    a = torch.softmax(torch.randn(5, 9, dtype=torch.float64), -1)
    closed = (a.pow(2).sum(-1) - 2 * a.pow(3).sum(-1) + a.pow(2).sum(-1).pow(2)).sqrt()
    assert torch.allclose(closed, jacobian_norm_direct(a))


def test_softmax_shift_invariance():
    assert shift_invariance() < 1e-12


def test_document_mask_stops_leaks():
    r = reach.packing_leak([5, 7, 4])
    assert r["causal only"] > 1e-3
    assert r["causal AND same document"] == 0.0


def test_padding_and_causal_compose():
    assert reach.padding_and_causal([7, 3, 10], n=10) < 1e-5


@pytest.mark.parametrize("w,depth", [(2, 3), (4, 2), (8, 4)])
def test_stacked_windows_reach_depth_times_w_minus_one(w, depth):
    assert reach.farthest_reach(64, w, depth) == depth * (w - 1)
    assert reach.gradient_reach(64, w, depth) == depth * (w - 1)


def test_hops_for_patterns():
    n = 128
    assert reach.hops_to_full(lambda _: reach.strided(n, 16)) == 2
    assert reach.hops_to_full(lambda _: reach.local_global(n, 8, 1)) == 2
    assert reach.hops_to_full(lambda _: reach.block_diagonal(n, 16)) is None
    assert reach.hops_to_full(lambda _: reach.causal(n)) == 1


def test_grid_patterns():
    r, c = reach.axial(8, 8)
    assert reach.hops_to_full(lambda i: r if i % 2 == 0 else c) == 2
    assert reach.hops_to_full(lambda _: reach.grid_windows(8, 8, 4), max_layers=8) is None
    assert reach.hops_to_full(lambda i: reach.grid_windows(8, 8, 4, shift=2 * (i % 2)), max_layers=8) is not None
