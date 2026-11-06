from math import comb

import numpy as np
import pytest

from agenteval.ablation import COMPONENTS, lattice, marginal_deltas, shapley
from agenteval.agents import noisy
from agenteval.graders import grade_state
from agenteval.harness import run_episode
from agenteval.noise import measured, runs_needed_normal
from agenteval.oversight import escalation_sweep, harm_formula, p_star
from agenteval.reliability import (empirical_h50, first_failure_steps, h50_exact, pass_at_k, pass_hat_k,
                                   second_order)
from agenteval.world import cancel_and_refund, initial_state


def test_pass_hat_k_estimator_is_unbiased():
    n, p, k = 10, 0.7, 3
    expect = sum(comb(n, c) * p ** c * (1 - p) ** (n - c) * pass_hat_k(n, c, k) for c in range(n + 1))
    assert expect == pytest.approx(p ** k)
    expect = sum(comb(n, c) * p ** c * (1 - p) ** (n - c) * pass_at_k(n, c, k) for c in range(n + 1))
    assert expect == pytest.approx(1 - (1 - p) ** k)


def test_jensen_and_bimodal_limit():
    ps = np.random.default_rng(0).beta(3, 2, 5000)
    for k in (2, 4, 8):
        assert np.mean(ps ** k) >= ps.mean() ** k
    bimodal = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    assert np.mean(bimodal ** 8) == bimodal.mean()
    narrow = np.random.default_rng(1).beta(60, 40, 20000)
    assert np.mean(narrow ** 4) == pytest.approx(second_order(narrow.mean(), narrow.var(), 4), rel=0.01)


def test_noisy_agent_succeeds_with_its_competence():
    rng = np.random.default_rng(0)
    task, s0 = cancel_and_refund("A100"), initial_state()
    wins = sum(grade_state(run_episode(noisy(0.6), task, s0, rng=rng)) for _ in range(2000))
    assert wins / 2000 == pytest.approx(0.6, abs=0.035)


def test_halving_hazard_and_repair_double_h50():
    rng = np.random.default_rng(0)
    assert h50_exact(0.01) / h50_exact(0.02) == pytest.approx(2.0, rel=0.01)
    assert h50_exact(0.02, rho=0.5) == pytest.approx(h50_exact(0.01))
    assert empirical_h50(first_failure_steps(0.02, 100_000, rng)) == pytest.approx(h50_exact(0.02), abs=1.5)


def test_grader_floor_ceiling_and_cost_ratios():
    assert measured(0.0, 0.08, 0.15) == pytest.approx(0.08)
    assert measured(1.0, 0.08, 0.15) == pytest.approx(0.85)
    base = runs_needed_normal(0.5, 0.6, 0, 0)
    assert runs_needed_normal(0.5, 0.6, 0.08, 0.15) / base == pytest.approx(0.77 ** -2, rel=0.03)
    assert runs_needed_normal(0.5, 0.6, 0.20, 0.25) / base == pytest.approx(0.55 ** -2, rel=0.03)


def test_p_star_values():
    assert p_star(1, 100) == pytest.approx(100 / 101)
    assert 0.97 < p_star(1, 100)
    assert p_star(1, 0) == 0
    assert p_star(1, 100, c_r=0.25) == pytest.approx(0.2)


def test_escalation_has_an_interior_optimum():
    e = np.linspace(0, 1, 101)
    h = harm_formula(e, 0.1)
    assert 0 < e[h.argmin()] < 1 and h[-1] > h[0]
    sweep = escalation_sweep([0, 0.2, 1.0], 50_000, np.random.default_rng(0))
    assert sweep[1]["harm"] < sweep[0]["harm"] < sweep[2]["harm"]


def test_ablation_lattice_shapley_sums_and_order_matters():
    lat = lattice(n=20_000)
    sh = shapley(lat)
    assert sum(sh.values()) == pytest.approx(lat[frozenset(COMPONENTS)] - lat[frozenset()])
    a = dict(marginal_deltas(lat, COMPONENTS))["retrieval"]
    b = dict(marginal_deltas(lat, ("verifier",) + tuple(c for c in COMPONENTS if c != "verifier")))["retrieval"]
    assert a > 0.02 and b == pytest.approx(0.0)
