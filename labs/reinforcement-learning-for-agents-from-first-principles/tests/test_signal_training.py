import math

import numpy as np
import pytest
import torch

from rlfirst.credit import (CreditConfig, Rollouts, anchor_advantages, episode_advantages, rollout as grid_rollout,
                            step_returns, train as train_grid, transition_table, worked_example)
from rlfirst.horizon import (group_size_needed, h50, h50_with_recovery, mc_informative_horizon,
                             mc_success_with_recovery, p_informative)
from rlfirst.lock import LockConfig, advantages, horizon_at, init_logits, train as train_lock
from rlfirst.rollouts import async_run, sync_throughput, sync_utilisation


# -- horizon arithmetic ---------------------------------------------------------------------------------------
def test_p_informative_endpoints_and_peak():
    assert p_informative(0.0, 8) == 0 and p_informative(1.0, 8) == 0
    ps = np.linspace(0, 1, 101)
    assert ps[np.argmax(p_informative(ps, 8))] == pytest.approx(0.5)


def test_post_numbers_for_q_093():
    assert 0.93 ** 10 == pytest.approx(0.484, abs=1e-3)
    assert 0.93 ** 30 == pytest.approx(0.113, abs=1e-3)
    assert 0.93 ** 50 == pytest.approx(0.0266, abs=1e-4)
    assert 1 / p_informative(0.93 ** 100, 8) == pytest.approx(177.7, abs=0.5)   # not "one in forty"


def test_monte_carlo_matches_formula():
    rng = np.random.default_rng(0)
    assert mc_informative_horizon(0.93, 30, 8, 40000, rng) == pytest.approx(p_informative(0.93 ** 30, 8), abs=0.01)


def test_group_size_grows_like_inverse_p():
    g50, g100 = group_size_needed(0.93, 50, 0.5), group_size_needed(0.93, 100, 0.5)
    assert p_informative(0.93 ** 50, g50) >= 0.5 > p_informative(0.93 ** 50, g50 - 1)
    assert g100 / g50 == pytest.approx(0.93 ** -50, rel=0.1)


def test_h50_and_recovery():
    assert h50(0.93) == pytest.approx(9.55, abs=0.01)
    H = round(h50_with_recovery(0.93, 0.8))
    assert mc_success_with_recovery(0.93, 0.8, H, 20000, np.random.default_rng(0)) == pytest.approx(0.5, abs=0.02)


# -- the lock -------------------------------------------------------------------------------------------------
def test_lock_init_and_schedule():
    cfg = LockConfig(H=6, n_tasks=3)
    codes = torch.randint(0, cfg.K, (3, 6))
    p = torch.softmax(init_logits(cfg, codes), -1).gather(2, codes[..., None])
    assert torch.allclose(p, torch.full_like(p, 0.8))
    cfg = LockConfig(H=40, curriculum=(5, 5, 30))
    assert [horizon_at(cfg, t) for t in (0, 29, 30, 209, 210, 500)] == [5, 5, 10, 35, 40, 40]


def test_zero_variance_group_has_zero_advantage():
    a = advantages(torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]))
    assert torch.all(a[0] == 0) and a[1, 1] > 0 > a[1, 0]


def test_short_lock_learns_long_lock_does_not():
    short, _, _ = train_lock(LockConfig(H=5, iters=40))
    long, _, _ = train_lock(LockConfig(H=40, iters=40))
    assert short.rows[-1]["target_sampled_success"] > 0.9
    assert long.rows[-1]["target_sampled_success"] < 0.01


def test_filtering_saves_backward_work_without_changing_the_run():
    a, la, _ = train_lock(LockConfig(H=20, iters=20))
    b, lb, _ = train_lock(LockConfig(H=20, iters=20, filter_zero=True))
    assert torch.allclose(la, lb, atol=1e-5)
    assert sum(r["backward_rollouts"] for r in b.rows) < sum(r["backward_rollouts"] for r in a.rows)


# -- step-level credit ----------------------------------------------------------------------------------------
def test_transition_table_walls():
    nxt = transition_table(3)
    assert nxt[0, 0] == 0 and nxt[0, 3] == 1 and nxt[0, 1] == 3 and nxt[4, 4] == 4


def test_worked_example_reload_flips_sign():
    reload = next(r for r in worked_example() if r["action"] == "reload" and r["rollout"] == "A")
    assert reload["A_traj"] > 0 > reload["A_step"] and reload["A_two_level"] < 0


def test_step_returns_discount_from_the_end():
    ro = Rollouts(torch.tensor([[0, 1, 2]]), torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool),
                  torch.tensor([3]), torch.tensor([1.0]))
    assert torch.allclose(step_returns(ro, 0.5), torch.tensor([[0.25, 0.5, 1.0]]))


def test_anchor_groups_only_shared_states():
    ro = Rollouts(torch.tensor([[0, 1], [0, 2]]), torch.zeros(2, 2, dtype=torch.long),
                  torch.ones(2, 2, dtype=torch.bool), torch.tensor([2, 2]), torch.tensor([1.0, 0.0]))
    adv, shared = anchor_advantages(ro, torch.zeros(2, dtype=torch.long), 0.9)
    assert shared == 0.5 and adv[0, 1] == 0 and adv[0, 0] > 0 > adv[1, 0]


def test_rollout_stops_at_goal():
    logits = torch.zeros(9, 5)
    logits[:, 3] = 50.0                                  # always right
    logits[2, 1] = 100.0                                 # then down
    logits[5, 1] = 100.0
    ro = grid_rollout(logits, transition_table(3), torch.zeros(2, dtype=torch.long), 8, 10, torch.Generator())
    assert ro.reward.tolist() == [1.0, 1.0] and ro.length.tolist() == [4, 4]


def test_episode_advantages_per_group():
    adv = episode_advantages(torch.tensor([1.0, 0.0, 1.0, 1.0]), torch.tensor([0, 0, 1, 1]))
    assert adv[0] > 0 > adv[1] and adv[2] == adv[3] == 0


def test_anchor_credit_learns_short_paths():
    rows = train_grid(CreditConfig(n=5, T=16, groups=4, lr=1.0, iters=60, mode="anchor"))
    assert rows[-1]["success"] > 0.9 and rows[-1]["wasted_frac"] < rows[0]["wasted_frac"]


# -- rollout engine -------------------------------------------------------------------------------------------
def test_sync_utilisation_falls_with_N_and_sigma():
    rng = np.random.default_rng(0)
    assert sync_utilisation(0.0, 64, 10, rng) == pytest.approx(1.0)
    u = [sync_utilisation(1.0, N, 500, rng) for N in (8, 64)]
    assert u[0] > u[1] and sync_utilisation(0.5, 64, 500, rng) > u[1]


def test_async_reaches_the_busy_worker_ceiling_and_k_bounds_age():
    rng = np.random.default_rng(0)
    a = async_run(1.0, 32, 32, None, 500, rng)
    assert a["throughput"] == pytest.approx(32 / math.exp(0.5), rel=0.08)
    assert a["throughput"] > 3 * sync_throughput(1.0, 32, 500, rng)
    b = async_run(1.0, 32, 32, 2, 500, rng)
    assert b["age_max"] <= 2 and b["dropped_frac"] > 0
