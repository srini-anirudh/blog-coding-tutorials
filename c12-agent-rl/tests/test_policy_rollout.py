import torch

from agentrl.batching import collate
from agentrl.bc import bc_loss, expert_demos, train_bc
from agentrl.codec import WordCodec, encode_trajectory
from agentrl.env import FileSystemEnv
from agentrl.policy import PolicyConfig, TinyPolicy, token_logprobs
from agentrl.rollout import rollout_batch
from agentrl.rollout_naive import rollout_one
from agentrl.tasks import sample_task

codec = WordCodec()


def tiny(seed: int = 0) -> TinyPolicy:
    torch.manual_seed(seed)
    return TinyPolicy(PolicyConfig(codec.vocab_size, d_model=32, n_layers=1, n_heads=2)).eval()


def test_shapes_and_param_count():
    pol = tiny()
    out = pol(torch.randint(0, codec.vocab_size, (3, 17)))
    assert out.logits.shape == (3, 17, codec.vocab_size) and out.values.shape == (3, 17)


def test_causality_future_tokens_do_not_change_past_logits():
    pol = tiny()
    x = torch.randint(0, codec.vocab_size, (2, 20))
    y = x.clone()
    y[:, 12:] = torch.randint(0, codec.vocab_size, (2, 8))
    torch.testing.assert_close(pol(x).logits[:, :12], pol(y).logits[:, :12])


def test_right_padding_needs_no_mask():
    pol = tiny()
    x = torch.randint(6, codec.vocab_size, (1, 10))
    padded = torch.cat([x, torch.zeros(1, 5, dtype=torch.long)], 1)
    torch.testing.assert_close(pol(x).logits, pol(padded).logits[:, :10])


def test_codec_roundtrip_and_trajectory_layout():
    text = "edit db/main port 42"
    assert codec.decode(codec.encode(text)) == text
    demo = expert_demos(1, (2,), codec, seed=3)[0]
    acts = [codec.decode([t]) for t, m in zip(demo.tokens, demo.action_mask) if m]
    assert acts[0] == "search" and acts[-1] == "<eos>" and acts[-2] == "submit"
    assert all(demo.tokens[p] == codec.act for p in demo.state_positions)
    assert len(demo.tokens) == len(demo.action_mask) == len(demo.step_of_token)


def test_naive_and_batched_rollouts_agree_greedy():
    pol = tiny()
    envs = [FileSystemEnv(sample_task(1 + i % 3, i)) for i in range(6)]
    batched = rollout_batch(pol, codec, envs, temperature=0)
    naive = [rollout_one(pol, codec, FileSystemEnv(sample_task(1 + i % 3, i)), temperature=0) for i in range(6)]
    for a, b in zip(batched, naive):
        assert a.tokens == b.tokens and a.action_mask == b.action_mask
        torch.testing.assert_close(torch.tensor(a.logprobs), torch.tensor(b.logprobs), atol=1e-5, rtol=0)


def test_rollout_logprobs_match_a_training_forward():
    """pi_old from sampling == pi_theta recomputed on the whole sequence: the GRPO ratio starts at exactly 1."""
    pol = tiny(1)
    torch.manual_seed(0)
    trajs = rollout_batch(pol, codec, [FileSystemEnv(sample_task(2, i)) for i in range(8)], temperature=1.0)
    batch = collate(trajs, codec.pad)
    logp = token_logprobs(pol(batch.tokens).logits, batch.tokens)
    torch.testing.assert_close(logp * batch.action_mask, batch.old_logprobs, atol=1e-5, rtol=0)


def test_rollout_tokens_equal_reencoded_text():
    """With a word codec the sampled ids and re-tokenised text agree (a BPE tokenizer gives no such promise)."""
    pol = tiny(2)
    torch.manual_seed(0)
    for tr in rollout_batch(pol, codec, [FileSystemEnv(sample_task(1, i)) for i in range(4)]):
        if all(s.action and "<" not in s.action for s in tr.steps) and tr.tokens[-1] == codec.eos:
            again = encode_trajectory(type(tr)(tr.task_text, tr.level, tr.seed, steps=tr.steps), codec)
            assert again.tokens == tr.tokens


def test_forced_steps_are_not_trained_on():
    pol = tiny()
    tr = rollout_batch(pol, codec, [FileSystemEnv(sample_task(1, 0))], forced=[["list db", "read db/main"]],
                       temperature=0)[0]
    assert tr.steps[0].forced and tr.steps[1].forced
    first_free = tr.state_positions[2]
    assert sum(tr.action_mask[:first_free + 1]) == 0


def test_bc_loss_drops_and_save_load_roundtrip(tmp_path):
    pol = tiny().train()
    demos = expert_demos(64, (1,), codec)
    before = bc_loss(pol, collate(demos, codec.pad)).item()
    train_bc(pol, demos, steps=150, batch_size=32, lr=3e-3)
    after = bc_loss(pol, collate(demos, codec.pad)).item()
    assert after < 0.5 * before
    pol.save(tmp_path / "p.pt")
    again = TinyPolicy.load(tmp_path / "p.pt").eval()
    x = collate(demos[:2], codec.pad).tokens
    torch.testing.assert_close(pol.eval()(x).logits, again(x).logits)
