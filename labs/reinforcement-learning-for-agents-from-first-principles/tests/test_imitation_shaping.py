import numpy as np
import pytest

from rlfirst.imitation import (STAY, Table, bc_bound, expert_action, fit_bc, fit_dagger, learner_action,
                               loglog_slope, regret, rollout)
from rlfirst.shaping import (GOAL, PAGES, REWARDS, cycle_gain, greedy_rollout, make_potential, q_learning,
                             simple_cycles, value_iteration)


def test_expert_steps_toward_zero_and_has_zero_cost():
    assert list(expert_action(np.array([-3, 0, 2]))) == [2, STAY, 0]
    _, cost = rollout(expert_action, 50, 10, np.random.default_rng(0))
    assert cost.sum() == 0


def test_clean_demos_only_label_the_start_state():
    t = fit_bc(30, 32, np.random.default_rng(0))
    assert t.n_states == 1 and t.has(np.array([0]))[0]


def test_learner_error_rate_on_labelled_states_is_eps():
    t = Table(5)
    t.add(np.array([0]))
    a = learner_action(t, np.zeros(200000, dtype=np.int64), 0.1, np.random.default_rng(0))
    assert abs((a != STAY).mean() - 0.1) < 0.005


def test_cliff_is_absorbing():
    _, cost = rollout(lambda x: np.full_like(x, 2), 10, 1, np.random.default_rng(0), cliff=1)
    assert cost[0] == 10                        # walked to 2 and stayed there


def test_bc_is_quadratic_and_dagger_linear_on_the_cliff():
    eps, Ts = 0.002, [10, 20, 50, 100]
    bc, dg = [], []
    for T in Ts:
        rng = np.random.default_rng(T)
        bc.append(regret(fit_bc(T, T + 2, rng, cliff=1), T, eps, rng, n=4000, cliff=1)[0])
        dg.append(regret(fit_dagger(T, T + 2, eps, rng, cliff=1), T, eps, rng, n=4000, cliff=1)[0])
    assert loglog_slope(Ts, bc) > 1.7
    assert 0.8 < loglog_slope(Ts, dg) < 1.3
    assert all(b <= bc_bound(eps, T) * 1.2 for b, T in zip(bc, Ts))


def test_sparse_and_potential_share_the_optimal_path():
    p_sparse, _, ok1 = greedy_rollout(REWARDS["sparse"], 0.99, 30)
    p_pot, _, ok2 = greedy_rollout(REWARDS["potential"], 0.99, 30)
    assert ok1 and ok2 and p_sparse == p_pot and len(p_sparse) == len(PAGES)


def test_milestone_reward_is_farmed_by_a_loop():
    path, ret, _ = greedy_rollout(REWARDS["milestone"], 0.99, 30)
    assert len(path) - 1 == 29 and ret > 1.0          # loops product <-> cart, checks out on the last step
    assert path.count("cart") > 10
    assert q_learning(REWARDS["milestone"], 0.99, 30, 2000)[0] == 0.0
    assert q_learning(REWARDS["sparse"], 0.99, 30, 2000) == (1.0, 5.0)


def test_cycle_gain_is_zero_for_potential_positive_for_milestone():
    cycles = simple_cycles()
    assert len(cycles) == 8
    assert all(cycle_gain(REWARDS["potential"], c) == pytest.approx(0.0) for c in cycles)
    assert max(cycle_gain(REWARDS["milestone"], c) for c in cycles) == pytest.approx(0.2)


def test_potential_shaping_telescopes():
    phi = {p: float(i) ** 2 for i, p in enumerate(PAGES)}
    r = make_potential(phi)
    path = [0, 1, 2, 3, 2, 3, 4, 5]
    shaped = sum(r(a, b, 1.0) for a, b in zip(path, path[1:]))
    assert shaped == pytest.approx(1.0 - phi["home"])       # sparse 1, plus Phi(goal) = 0 minus Phi(start)


def test_timeout_potential_can_change_the_policy():
    _, q_sparse = value_iteration(REWARDS["sparse"], 0.99, 30)
    phi = {p: 0.0 for p in PAGES}
    phi["home"] = 5.0                                     # staying home at time-out keeps gamma^H * 5
    _, q_bad = value_iteration(make_potential(phi, zero_at_timeout=False), 0.99, 30)
    _, q_good = value_iteration(make_potential(phi, zero_at_timeout=True), 0.99, 30)
    best = lambda q: max(q, key=q.get)                    # noqa: E731
    assert best(q_good[0]) == best(q_sparse[0]) == "open_search"
    assert best(q_bad[0]) == "reload"
    assert GOAL == PAGES.index("checkout")
