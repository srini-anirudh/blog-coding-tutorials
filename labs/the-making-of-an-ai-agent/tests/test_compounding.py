import random

from agentcraft.compounding import BROKEN, exact_cost, expert, fit_bc, fit_dagger, mean_cost, quadratic_approx


def test_exact_sum_is_quadratic_for_small_eps():
    assert abs(exact_cost(1e-4, 20) - quadratic_approx(1e-4, 20)) / quadratic_approx(1e-4, 20) < 0.01


def test_expert():
    assert expert(0) == 0 and expert(2) == -1 and expert(-1) == 1 and expert(BROKEN) == 0


def test_bc_matches_absorbing_formula_and_dagger_is_linear():
    eps, H = 0.05, 20
    rng = random.Random(0)
    bc = fit_bc(H, 20, rng)
    assert bc(1) == 0                                   # never saw d = 1, so it does not step back
    assert abs(mean_cost(bc, H, eps, 5000, 1) - exact_cost(eps, H)) < 0.4
    dagger = fit_dagger(H, eps, 20, 3, rng)
    assert dagger(1) == -1 and dagger(-1) == 1
    assert abs(mean_cost(dagger, H, eps, 5000, 2) - eps * H) < 0.15


def test_irreversible_mistakes_defeat_dagger():
    eps, H = 0.05, 20
    dagger = fit_dagger(H, eps, 20, 3, random.Random(0), irreversible=1.0)
    assert abs(mean_cost(dagger, H, eps, 5000, 3, irreversible=1.0) - exact_cost(eps, H)) < 0.4
