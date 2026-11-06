import random

import torch

from agentcraft.credit import (ADVANCE, ChainConfig, Episode, anchor_advantages, episode_advantages, make_roles,
                               run_episode, success_probability, train)


def test_episode_advantages_zero_when_rewards_equal():
    group = [Episode(0, [0], [0], [(0, 0)], 1.0) for _ in range(4)]
    assert episode_advantages(group) == [0.0] * 4


def test_anchor_groups_rank_actions_from_the_same_state():
    # Both trajectories succeed; the first wastes a step at stage 0, so its stage-0 actions get lower returns.
    slow = Episode(0, [0, 0, 1], [2, 0, 0], [(0, 0), (0, 0), (0, 1)], 1.0)
    fast = Episode(0, [0, 1], [0, 0], [(0, 0), (0, 1)], 1.0)
    a_e = episode_advantages([slow, fast])
    a_s, coverage = anchor_advantages([slow, fast], gamma=0.9)
    assert a_e == [0.0, 0.0]                             # GRPO sees no difference
    assert a_s[0][0] < 0 < a_s[1][0]                     # GiGPO does
    assert coverage == 1.0


def test_unique_states_give_no_step_signal():
    group = [Episode(0, [0], [0], [(0, 0, random.random())], float(i % 2)) for i in range(4)]
    a_s, coverage = anchor_advantages(group, 0.9)
    assert coverage == 0.0 and all(v == [0.0] for v in a_s)


def test_exact_success_matches_sampling():
    cfg = ChainConfig(n_tasks=2)
    roles = make_roles(cfg, 0)
    probs = torch.full((cfg.n_tasks, cfg.stages, cfg.actions), 1 / cfg.actions)
    exact = success_probability(probs, roles, cfg)[0].item()
    rng = random.Random(0)
    mc = sum(run_episode(0, probs, roles, cfg, rng).reward for _ in range(20000)) / 20000
    assert abs(exact - mc) < 0.01
    perfect = torch.tensor([[[1.0 if r == ADVANCE else 0.0 for r in st] for st in task] for task in roles])
    assert torch.allclose(success_probability(perfect, roles, cfg), torch.ones(cfg.n_tasks))


def test_gigpo_removes_noops_faster():
    g = train("grpo", 60, 0, eval_at=(60,), log=lambda s: None)["history"][-1]
    s = train("gigpo", 60, 0, eval_at=(60,), log=lambda s: None)["history"][-1]
    assert s["noop_prob"] < g["noop_prob"]
