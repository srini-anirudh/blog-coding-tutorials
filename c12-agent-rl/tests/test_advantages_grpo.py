import copy

import numpy as np
import pytest
import torch

from agentrl.advantages import gae, group_relative, to_tokens
from agentrl.batching import collate
from agentrl.codec import WordCodec, encode_trajectory
from agentrl.env import FileSystemEnv, run_episode
from agentrl.expert import ScriptedExpert
from agentrl.grpo import GRPOConfig, grpo_loss, step_advantages, update
from agentrl.online import Curriculum, TaskMix, TaskSampler, group_rollouts, group_stats
from agentrl.policy import PolicyConfig, TinyPolicy, token_logprobs
from agentrl.rewards import activity, outcome, potential
from agentrl.rollout import rollout_batch
from agentrl.tasks import sample_task
from agentrl.trajectory import Step, Trajectory

codec = WordCodec()


def tiny(seed: int = 0) -> TinyPolicy:
    torch.manual_seed(seed)
    return TinyPolicy(PolicyConfig(codec.vocab_size, d_model=32, n_layers=1, n_heads=2)).eval()


def test_group_relative_matches_numpy_reference():
    rewards = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0], dtype=np.float32)
    groups = np.array([0] * 4 + [1] * 4 + [2] * 4)
    ours = group_relative(torch.tensor(rewards), torch.tensor(groups), eps=1e-4).numpy()
    ref = np.zeros_like(rewards)
    for g in range(3):
        r = rewards[groups == g]
        ref[groups == g] = (r - r.mean()) / (r.std(ddof=1) + 1e-4)   # torch.std is the unbiased (ddof=1) std
    np.testing.assert_allclose(ours, ref, atol=1e-5)


def test_group_relative_properties():
    r = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
    a = group_relative(r, torch.zeros(8, dtype=torch.long))
    assert abs(a.sum()) < 1e-5 and a[0] > 0 > a[1]
    # one winner of eight gets a large advantage; seven of eight gets a small one: rarity is rewarded
    b = group_relative(1 - r, torch.zeros(8, dtype=torch.long))
    assert a[0] > b[1] > 0
    # the same success in an all-success or all-fail group teaches nothing
    assert torch.all(group_relative(torch.ones(4), torch.zeros(4, dtype=torch.long)) == 0)
    assert torch.all(group_relative(torch.zeros(4), torch.zeros(4, dtype=torch.long)) == 0)
    # the mean-only variant keeps the sign and drops the amplification
    c = group_relative(r, torch.zeros(8, dtype=torch.long), scale=False)
    assert c[0] == pytest.approx(0.875) and c[1] == pytest.approx(-0.125)


def test_gae_matches_explicit_sum_and_special_cases():
    r, v = [0.0, 0.0, 1.0], [0.5, 0.2, 0.9]
    g, lam = 0.9, 0.8
    d = [r[0] + g * v[1] - v[0], r[1] + g * v[2] - v[1], r[2] - v[2]]
    expect = [d[0] + g * lam * d[1] + (g * lam) ** 2 * d[2], d[1] + g * lam * d[2], d[2]]
    np.testing.assert_allclose(gae(r, v, g, lam), expect)
    # lambda = 1, gamma = 1: Monte Carlo return minus the baseline
    np.testing.assert_allclose(gae(r, v, 1.0, 1.0), [1 - 0.5, 1 - 0.2, 1 - 0.9])
    # lambda = 0: exactly the one-step TD error
    np.testing.assert_allclose(gae(r, v, 1.0, 0.0), [0.2 - 0.5, 0.9 - 0.2, 1 - 0.9])


def expert_traj(level=3, seed=4, forced=None) -> Trajectory:
    task = sample_task(level, seed)
    return encode_trajectory(run_episode(FileSystemEnv(task), ScriptedExpert(task.text), forced=forced), codec)


def test_to_tokens_puts_step_advantage_on_its_action_tokens_only():
    tr = expert_traj()
    b = collate([tr], codec.pad)
    adv = to_tokens([[1.0, 2.0, 3.0, 4.0]], b.step_of_token)
    assert ((adv != 0) == (b.action_mask != 0)).all()
    # token 0 of "search KEY <eos>" is step 0
    first_action = tr.state_positions[0] + 1
    assert adv[0, first_action] == 1.0 and adv[0, tr.state_positions[1] + 1] == 2.0


def test_potential_shaping_telescopes_to_the_outcome():
    """Episode total of potential == outcome, so trajectory-level GRPO cannot tell them apart."""
    for level, seed in [(1, 1), (2, 2), (3, 3), (5, 5)]:
        tr = expert_traj(level, seed)
        assert sum(potential(tr)) == pytest.approx(sum(outcome(tr)))
        assert any(r > 0 for r in potential(tr)[:-1])         # but credit does arrive early
    fail = Trajectory("t", 2, 0, steps=[Step("search port", "found a", progress=0.5),
                                        Step("read a", "x", progress=0.0)], success=False)
    assert sum(potential(fail)) == pytest.approx(0.0)


def test_activity_reward_pays_more_for_looping_than_for_solving():
    solved = expert_traj(2, 1)
    task = sample_task(2, 1)
    path = sorted(task.files)[0]
    loop = run_episode(FileSystemEnv(task), lambda _: f"read {path}")
    assert sum(activity(loop)) > sum(activity(solved)) and not loop.success and solved.success


def make_group(policy, n_tasks=2, G=4, cfg=None):
    cfg = cfg or GRPOConfig()
    sampler = TaskSampler(TaskMix({1: 1.0}))
    return group_rollouts(policy, codec, sampler.sample(n_tasks), G, cfg, "cpu")


def test_group_rollouts_share_task_and_group_id():
    trajs = make_group(tiny(), 3, 4)
    assert len(trajs) == 12 and len({t.group for t in trajs}) == 3
    assert len({t.task_text for t in trajs[:4]}) == 1


def test_first_ratio_is_one_so_first_surrogate_is_weighted_mean():
    pol = tiny(3)
    torch.manual_seed(1)
    trajs = make_group(pol)
    cfg = GRPOConfig()
    b = collate(trajs, codec.pad)
    logp = token_logprobs(pol(b.tokens).logits, b.tokens)
    torch.testing.assert_close(logp * b.action_mask, b.old_logprobs, atol=1e-5, rtol=0)
    adv = torch.ones_like(b.old_logprobs)
    loss, stats = grpo_loss(pol, b, adv, cfg)
    assert loss.item() == pytest.approx(-1.0, abs=1e-4) and stats["clip_frac"] == 0.0


def test_positive_advantage_raises_logprob_of_those_actions():
    pol = tiny(4).train()
    torch.manual_seed(2)
    trajs = make_group(pol)
    b = collate(trajs, codec.pad)
    before = token_logprobs(pol(b.tokens).logits, b.tokens).detach()
    opt = torch.optim.SGD(pol.parameters(), lr=0.1)
    loss, _ = grpo_loss(pol, b, b.action_mask.clone(), GRPOConfig())
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pol.parameters())
    opt.step()
    after = token_logprobs(pol(b.tokens).logits, b.tokens).detach()
    assert (after - before)[b.action_mask == 1].mean() > 0


def test_environment_tokens_get_no_gradient_signal():
    pol = tiny(5)
    torch.manual_seed(2)
    trajs = make_group(pol)
    b = collate(trajs, codec.pad)
    adv = torch.randn(b.tokens.shape)
    l1, _ = grpo_loss(pol, b, adv, GRPOConfig())
    adv2 = adv.clone()
    adv2[b.action_mask == 0] = 100.0        # advantages off the mask must not matter
    l2, _ = grpo_loss(pol, b, adv2, GRPOConfig())
    assert l1.item() == pytest.approx(l2.item())


def test_zero_variance_batch_produces_zero_policy_gradient():
    pol = tiny(6)
    torch.manual_seed(0)
    trajs = make_group(pol)                  # a random policy: every reward is 0
    assert not any(t.success for t in trajs)
    step_adv, _ = step_advantages(trajs, GRPOConfig())
    assert all(a == 0.0 for adv in step_adv for a in adv)
    assert group_stats(trajs, "outcome")["zero_var_groups"] == 1.0


@pytest.mark.parametrize("credit", ["trajectory", "action"])
def test_update_changes_parameters_and_reports_stats(credit):
    pol = tiny(7)
    torch.manual_seed(0)
    trajs = make_group(pol)
    trajs[0].success = True                  # one winner, so there is signal
    trajs[0].steps[-1].reward = 1.0
    cfg = GRPOConfig(credit=credit, minibatch=4, epochs=1)
    before = copy.deepcopy(pol.state_dict())
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-3)
    stats = update(pol, opt, trajs, cfg, pad=codec.pad, device="cpu")
    assert "loss" in stats and any((before[k] != v).any() for k, v in pol.state_dict().items())
    assert ("value_loss" in stats) == (credit == "action")


def test_kl_penalty_is_zero_for_the_reference_itself():
    pol = tiny(8)
    torch.manual_seed(0)
    trajs = make_group(pol)
    b = collate(trajs, codec.pad)
    _, stats = grpo_loss(pol, b, torch.zeros_like(b.old_logprobs), GRPOConfig(kl_beta=0.1), ref_policy=pol)
    assert stats["kl"] == pytest.approx(0.0, abs=1e-6)


def test_curriculum_advances_on_threshold_only():
    c = Curriculum([TaskMix({1: 1.0}), TaskMix({2: 1.0})], threshold=0.7, window=2)
    assert not c.record(0.9) and c.stage == 0
    assert c.record(0.8) and c.stage == 1 and c.mix.levels == {2: 1.0}
    assert not c.record(1.0) and not c.record(1.0) and c.stage == 1  # last stage stays


def test_forced_steps_excluded_from_token_advantages():
    pol = tiny(9)
    tr = rollout_batch(pol, codec, [FileSystemEnv(sample_task(1, 0))], forced=[["list db"]], temperature=0)[0]
    b = collate([tr], codec.pad)
    adv = to_tokens([[5.0] * tr.n_steps], b.step_of_token)
    assert adv[0, tr.state_positions[0] + 1] == 0.0
