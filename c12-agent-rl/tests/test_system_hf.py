import copy
import json

import pytest
import torch
from torch import nn

from agentrl.batching import collate
from agentrl.codec import WordCodec
from agentrl.env import FileSystemEnv, run_episode
from agentrl.expert import ScriptedExpert
from agentrl.grpo import GRPOConfig, update
from agentrl.hf_policy import smollm_policy, tiny_llama_policy
from agentrl.online import TaskMix, TaskSampler
from agentrl.policy import PolicyConfig, TinyPolicy, token_logprobs
from agentrl.rollout import rollout_batch
from agentrl.system import AgentRLSystem, EnvWorkers, SlowTool, SystemConfig, TrajectoryStore, Verifier, make_batches
from agentrl.tasks import sample_task
from agentrl.trajectory import Trajectory

codec = WordCodec()


def tiny(seed: int = 0) -> TinyPolicy:
    torch.manual_seed(seed)
    return TinyPolicy(PolicyConfig(codec.vocab_size, d_model=32, n_layers=1, n_heads=2)).eval()


def expert_trajectory(level=2, seed=1) -> Trajectory:
    task = sample_task(level, seed)
    tr = run_episode(FileSystemEnv(task), ScriptedExpert(task.text))
    tr.meta["env_seed"] = task.seed
    return tr


def test_trajectory_json_roundtrip(tmp_path):
    tr = rollout_batch(tiny(), codec, [FileSystemEnv(sample_task(1, 0))], temperature=1.0)[0]
    tr.group = 3
    store = TrajectoryStore(tmp_path / "t.jsonl")
    store.append([tr])
    again = store.load()[0]
    assert again == Trajectory.from_json(tr.to_json()) and again.tokens == tr.tokens and again.group == 3
    assert json.loads(tr.to_json())["steps"][0]["action"] == tr.steps[0].action
    assert store.load(min_version=1) == []


def test_verifier_catches_a_false_claim_and_a_false_denial():
    v = Verifier()
    good = expert_trajectory()
    assert v.verify(copy.deepcopy(good)).success and v.mismatches == 0
    lie = copy.deepcopy(good)
    lie.steps = lie.steps[-1:]                # only "submit", but it still claims success
    assert not v.verify(lie).success and v.mismatches == 1
    denied = copy.deepcopy(good)
    denied.success = False
    assert v.verify(denied).success and denied.steps[-1].reward == 1.0 and v.mismatches == 2


def test_verifier_agrees_with_rollout_under_failure_injection():
    v = Verifier()
    envs = [FileSystemEnv(sample_task(4, s), seed=s + 100) for s in range(20)]
    for env in envs:
        task = env.task
        tr = run_episode(env, ScriptedExpert(task.text))
        tr.meta["env_seed"] = env.seed
        v.verify(tr)
    assert v.checked == 20 and v.mismatches == 0


def test_make_batches_drops_stale_versions_and_lonely_trajectories():
    a = Trajectory("t", 1, 0, group=0, policy_version=5)
    b = Trajectory("t", 1, 0, group=0, policy_version=5)
    old = Trajectory("t", 1, 1, group=1, policy_version=2)
    alone = Trajectory("t", 1, 2, group=2, policy_version=5)
    assert make_batches([a, b, old, alone], current_version=5) == [a, b]
    assert len(make_batches([a, b, old], current_version=5, max_lag=3)) == 2  # group 1 has one member left


def test_env_workers_give_the_same_trajectories_as_inline():
    pol = tiny()
    envs = lambda: [FileSystemEnv(sample_task(2, i)) for i in range(4)]  # noqa: E731
    inline = rollout_batch(pol, codec, envs(), temperature=0, step_fn=EnvWorkers(0))
    threaded = rollout_batch(pol, codec, envs(), temperature=0, step_fn=EnvWorkers(3))
    assert [t.tokens for t in inline] == [t.tokens for t in threaded]


def make_system(path, seed=0) -> AgentRLSystem:
    pol = tiny(seed).train()
    return AgentRLSystem(pol, codec, TaskSampler(TaskMix({1: 1.0}), seed=seed), GRPOConfig(minibatch=8, epochs=1),
                         SystemConfig(tasks_per_iter=2, G=2), path)


def test_system_iteration_runs_all_stages(tmp_path):
    s = make_system(tmp_path / "t.jsonl")
    row = s.iteration()
    assert set(row["clock"]) == {"sample", "rollout", "store", "verify", "batch", "update"}
    assert row["version"] == 1 and len(s.store.load()) == 4 and s.verifier.checked == 4


def test_resume_gives_the_same_weights_as_an_uninterrupted_run(tmp_path):
    a = make_system(tmp_path / "a.jsonl")
    a.iteration()
    a.save(tmp_path / "ckpt.pt")
    a.iteration()
    expected = copy.deepcopy(a.policy.state_dict())

    b = make_system(tmp_path / "b.jsonl", seed=0)
    b.load(tmp_path / "ckpt.pt")
    b.iteration()
    for k, v in expected.items():
        torch.testing.assert_close(b.policy.state_dict()[k], v, atol=1e-6, rtol=0)


def test_slow_tool_latency_is_applied():
    env = SlowTool(sample_task(1, 0))
    env.latency = 0.0
    env.reset()
    assert env.step("submit")[2]


def test_tiny_llama_drops_in_for_rollout_and_update():
    """The trusted HF Llama architecture runs the same rollout and the same update, with no code change."""
    torch.manual_seed(0)
    pol = tiny_llama_policy(codec.vocab_size, hidden=32, layers=1, heads=2).eval()
    trajs = rollout_batch(pol, codec, [FileSystemEnv(sample_task(1, i)) for i in range(4)], temperature=1.0)
    for i, t in enumerate(trajs):
        t.group = i // 2
    b = collate(trajs, codec.pad)
    logp = token_logprobs(pol(b.tokens).logits, b.tokens)
    torch.testing.assert_close(logp * b.action_mask, b.old_logprobs, atol=1e-5, rtol=0)
    trajs[0].success = True
    trajs[0].steps[-1].reward = 1.0
    before = copy.deepcopy(pol.state_dict())
    update(pol, torch.optim.AdamW(pol.parameters(), lr=1e-3), trajs, GRPOConfig(minibatch=4, epochs=1),
           pad=codec.pad, device="cpu")
    assert any((before[k] != v).any() for k, v in pol.state_dict().items())


def test_tiny_policy_right_padding_contract_matches_llama():
    """Right padding is safe for our block and for HF Llama in the same way: causal attention never looks ahead."""
    torch.manual_seed(0)
    llama = tiny_llama_policy(codec.vocab_size, hidden=32, layers=1, heads=2).eval()
    x = torch.randint(6, codec.vocab_size, (2, 9))
    padded = torch.cat([x, torch.zeros(2, 4, dtype=torch.long)], 1)
    torch.testing.assert_close(llama(x).logits, llama(padded).logits[:, :9], atol=1e-5, rtol=1e-4)


@pytest.mark.network
def test_smollm_lora_runs_one_episode():
    pol, hf = smollm_policy()
    pol = pol.eval()
    trajs = rollout_batch(pol, hf, [FileSystemEnv(sample_task(1, 0))], temperature=1.0, max_action_tokens=8)
    assert trajs[0].n_steps > 0
    trainable = sum(p.numel() for p in pol.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pol.parameters())
    assert 0 < trainable < total * 0.1 and isinstance(pol.value_head, nn.Linear)
