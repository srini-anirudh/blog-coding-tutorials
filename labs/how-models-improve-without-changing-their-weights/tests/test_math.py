"""Closed forms from the post, checked against simulation or brute force."""

import itertools
import math

import numpy as np
import pytest

from ttc.allocate import closed_form, expected_solved, last_gain, stop_when_solved, uniform, water_fill
from ttc.cost import wide_vs_deep
from ttc.coverage import (beta_uncovered, coverage_one, fit_beta_binomial, half_coverage_k, loglog_slope,
                          pass_at_k, sample_population, simulate_coverage)
from ttc.selection import (fp_ceiling, majority_vote, modal_answer_correct, verifier_accuracy,
                           verifier_accuracy_mc)
from ttc.walk import accuracy_curve, negative_flips, sample_walks


# Coverage
def test_half_coverage_is_about_069_over_p():
    for p in (0.01, 0.001):
        assert half_coverage_k(p) == pytest.approx(0.69 / p, rel=0.01)
        assert coverage_one(p, half_coverage_k(p)) == pytest.approx(0.5)


@pytest.mark.parametrize("a", [0.35, 0.7, 1.4])
def test_beta_tail_slope_is_minus_a(a):
    ks = [1e3, 1e4, 1e5]
    assert loglog_slope(ks, [beta_uncovered(a, 3.0, k) for k in ks]) == pytest.approx(-a, abs=0.01)


def test_beta_formula_matches_monte_carlo():
    rng = np.random.default_rng(1)
    p = sample_population(0.7, 3.0, 50000, 0.0, rng)
    mc = 1 - simulate_coverage(p, [10, 100], rng)
    assert mc == pytest.approx([beta_uncovered(0.7, 3.0, 10), beta_uncovered(0.7, 3.0, 100)], abs=0.01)


def test_hopeless_fraction_is_a_floor():
    rng = np.random.default_rng(2)
    p = sample_population(0.7, 3.0, 50000, 0.2, rng)
    assert 1 - simulate_coverage(p, [10**6], rng)[0] == pytest.approx(0.2, abs=0.01)


def test_pass_at_k_matches_combinatorics():
    for n, c, k in [(10, 3, 1), (10, 3, 4), (20, 0, 5), (8, 7, 2)]:
        assert pass_at_k(n, c, k) == pytest.approx(1 - math.comb(n - c, k) / math.comb(n, k))


def test_beta_binomial_fit_recovers_shape():
    rng = np.random.default_rng(3)
    s = rng.binomial(64, rng.beta(0.7, 3.0, size=600))
    a, b = fit_beta_binomial(s, 64)
    assert a == pytest.approx(0.7, abs=0.2)


# Selection
def test_majority_vote_and_mode():
    rng = np.random.default_rng(0)
    assert majority_vote([1, 2, 2, None, None, None], rng) == 2
    assert majority_vote([None], rng) is None
    assert modal_answer_correct([5, 5, 3], 5) and not modal_answer_correct([5, 3], 5)


def test_post_ceilings():
    assert [round(fp_ceiling(0.05, f), 2) for f in (0, 0.01, 0.05, 0.2)] == [1.0, 0.84, 0.51, 0.21]


def test_verifier_formula_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for p, phi, k in [(0.05, 0.05, 16), (0.2, 0.01, 4), (0.02, 0.2, 64)]:
        assert verifier_accuracy(p, phi, k) == pytest.approx(verifier_accuracy_mc(p, phi, k, 40000, rng), abs=0.01)


def test_perfect_verifier_equals_coverage():
    assert verifier_accuracy(0.05, 0.0, [1, 10, 100]) == pytest.approx(coverage_one(0.05, [1, 10, 100]))


# Cost
def test_wide_vs_deep_bill():
    out = wide_vs_deep(1e8, 12, 768, 256, K=8, T=100)
    q = out["deep_over_wide"]
    assert q["weight_flops"] == q["kv_slots"] == q["kv_bytes"] == 1.0
    assert q["decode_steps"] == 8
    assert q["attention_flops"] == pytest.approx((800 + 1) / (100 + 1))


# Walk
def test_walk_monotone_without_pull_and_peaks_with_it():
    assert np.all(np.diff(accuracy_curve(200, 0.1, 0.0)) >= -1e-12)
    acc = accuracy_curve(200, 0.1, 0.01)
    assert acc.argmax() < 199 and acc[-1] < acc.max() - 0.05


def test_easy_problems_peak_earlier():
    assert accuracy_curve(200, 0.25, 0.004).argmax() < accuracy_curve(200, 0.05, 0.004).argmax()


def test_walk_sampling_matches_exact():
    rng = np.random.default_rng(0)
    w = sample_walks(20000, 100, 0.1, 0.01, 30.0, rng)
    assert (w[:, -1] == 1).mean() == pytest.approx(accuracy_curve(100, 0.1, 0.01)[-1], abs=0.015)
    assert negative_flips(sample_walks(2000, 100, 0.1, 0.0, 30.0, rng)) == 0.0


# Allocation
def test_greedy_is_optimal_on_small_instance():
    ps, B = [0.6, 0.2, 0.05], 6
    best = max(expected_solved(ps, a) for a in itertools.product(range(B + 1), repeat=3) if sum(a) == B)
    assert expected_solved(ps, water_fill(ps, B)) == pytest.approx(best)


def test_closed_form_and_peak():
    ps = np.geomspace(0.9, 0.002, 12)
    alloc = water_fill(ps, 400)
    lam = last_gain(ps, alloc)
    for p, b in zip(ps, alloc):
        assert abs(closed_form(p, lam, exact=True) - b) < 1.5
    assert closed_form(lam / 2, lam) == 0.0
    grid = np.linspace(lam * 1.01, 0.5, 20000)
    peak = grid[np.argmax([closed_form(p, lam) for p in grid])]
    assert peak == pytest.approx(math.e * lam, rel=0.01)
    assert expected_solved(ps, alloc) > expected_solved(ps, uniform(12, 400))


def test_stop_when_solved_never_wastes_on_sure_problems():
    rng = np.random.default_rng(0)
    assert stop_when_solved([1.0] * 5, 3, rng, trials=5) == 3
    assert stop_when_solved([1.0] * 5, 10, rng, trials=5) == 5
