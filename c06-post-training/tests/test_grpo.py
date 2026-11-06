"""Stage 4-7 checks: advantages, the clipped loss, token masks, KL, TRL equivalence, gradient noise, rollouts."""

import tempfile

import pytest
import torch
from datasets import Dataset

from posttrain.chat import load_tokenizer
from posttrain.generation import completion_mask_from_ids, left_pad, score_completions
from posttrain.grpo import GRPOLossConfig, group_advantages, grpo_loss
from posttrain.models import frozen_copy, load_policy, random_policy, tiny_llama
from posttrain.reinforce import estimator_weights, gradient_noise, reinforce_loss
from posttrain.rollout import RolloutEngine
from posttrain.tasks import ANSWER_RE, exact_reward, extract_answer, sample_problems


def test_group_advantages_by_hand():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    adv = group_advantages(rewards, group_size=4, eps=0.0).view(2, 4)
    assert adv.sum(1).abs().max() < 1e-6  # zero mean inside each group
    assert abs(adv.std(1)[0].item() - 1.0) < 1e-6  # unit std inside each group
    torch.testing.assert_close(adv[0], torch.tensor([1.5, -0.5, -0.5, -0.5]))


def test_all_same_reward_gives_zero_advantage():
    assert group_advantages(torch.ones(4), 4).abs().max() == 0
    assert group_advantages(torch.zeros(4), 4).abs().max() == 0


def test_group_advantages_match_trl_formula():
    from trl.trainer.utils import nanstd

    r = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    grouped = r.view(-1, 4)
    want = (grouped - grouped.mean(1, keepdim=True)) / (nanstd(grouped, dim=1).unsqueeze(1) + 1e-4)
    torch.testing.assert_close(group_advantages(r, 4), want.reshape(-1))


def test_first_inner_step_has_ratio_one_and_no_clipping():
    logps = torch.tensor([[-1.0, -2.0, -3.0]])
    loss, stats = grpo_loss(logps, logps.clone(), torch.tensor([1.0]), torch.ones(1, 3), cfg=GRPOLossConfig(beta=0.0))
    assert abs(stats["ratio_mean"] - 1.0) < 1e-6
    assert stats["clip_frac"] == 0
    assert abs(loss.item() + 1.0) < 1e-6  # -ratio * advantage


def test_clipping_removes_the_incentive_past_one_plus_eps():
    old = torch.zeros(1, 1)
    far = torch.tensor([[1.0]], requires_grad=True)  # ratio e > 1.2, advantage > 0
    loss, stats = grpo_loss(far, old, torch.tensor([1.0]), torch.ones(1, 1), cfg=GRPOLossConfig(beta=0.0))
    loss.backward()
    assert far.grad.abs().item() == 0
    assert stats["clip_frac"] == 1
    assert abs(loss.item() + 1.2) < 1e-6


def test_negative_advantage_is_never_clipped_upward():
    """With A < 0 and ratio > 1 + eps, min() picks the unclipped term: moving further keeps being penalised."""
    far = torch.tensor([[1.0]], requires_grad=True)
    loss, _ = grpo_loss(far, torch.zeros(1, 1), torch.tensor([-1.0]), torch.ones(1, 1), cfg=GRPOLossConfig(beta=0.0))
    loss.backward()
    assert far.grad.item() > 0


def test_padding_tokens_do_not_change_the_loss():
    logps = torch.tensor([[-1.0, -2.0, -50.0], [-1.5, -0.5, -0.1]])
    old = logps - 0.05
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    a, _ = grpo_loss(logps, old, adv, mask, cfg=GRPOLossConfig(beta=0.0))
    changed = logps.clone()
    changed[0, 2] = -7.0  # a padded position
    b, _ = grpo_loss(changed, old, adv, mask, cfg=GRPOLossConfig(beta=0.0))
    assert a.item() == b.item()


def test_grpo_loss_is_length_normalised_but_dapo_is_token_weighted():
    logps = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    adv = torch.tensor([1.0, -3.0])
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
    grpo, _ = grpo_loss(logps, logps, adv, mask, cfg=GRPOLossConfig(beta=0.0, loss_type="grpo"))
    dapo, _ = grpo_loss(logps, logps, adv, mask, cfg=GRPOLossConfig(beta=0.0, loss_type="dapo"))
    assert abs(grpo.item() - 1.0) < 1e-6  # (-1 + 3) / 2 sequences
    assert abs(dapo.item() + 0.2) < 1e-6  # (-4 + 3) / 5 tokens


def test_kl_is_zero_at_the_reference_and_positive_elsewhere():
    logps = torch.tensor([[-1.0, -2.0]])
    same, s = grpo_loss(logps, logps, torch.zeros(1), torch.ones(1, 2), ref_logps=logps, cfg=GRPOLossConfig(beta=1.0))
    assert s["kl"] == 0 and same.item() == 0
    _, moved = grpo_loss(logps, logps, torch.zeros(1), torch.ones(1, 2), ref_logps=logps - 1, cfg=GRPOLossConfig(beta=1.0))
    assert moved["kl"] > 0


def test_completion_mask_stops_after_first_eos():
    ids = torch.tensor([[5, 2, 7, 2], [5, 6, 7, 8]])
    mask, truncated = completion_mask_from_ids(ids, eos_id=2)
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert truncated.tolist() == [False, True]


def test_score_completions_matches_a_full_forward():
    model = tiny_llama()
    ids, mask = left_pad([[5, 6, 7], [8, 9]], pad_id=0, multiple=1)
    comp = torch.tensor([[10, 11, 2], [12, 2, 0]])
    cmask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    with torch.no_grad():
        got = score_completions(model, ids, mask, comp, cmask)
        full = model(input_ids=torch.cat([ids, comp], 1), attention_mask=torch.cat([mask, cmask], 1)).logits.log_softmax(-1)
    assert abs(got[0, 1].item() - full[0, 3, 11].item()) < 1e-4
    assert abs(got[1, 0].item() - full[1, 2, 12].item()) < 1e-4


@pytest.mark.parametrize("loss_type", ["grpo", "dapo"])
@pytest.mark.network
def test_matches_trl_grpo_loss_on_the_same_tensors(loss_type):
    """Feed one fixed batch to our loss and to trl.GRPOTrainer._compute_loss."""
    from trl import GRPOConfig, GRPOTrainer

    tok = load_tokenizer()
    tok.padding_side = "left"
    model = tiny_llama(vocab_size=len(tok), seed=0).eval()
    ref = tiny_llama(vocab_size=len(tok), seed=1).eval()
    data = Dataset.from_list([{"prompt": [{"role": "user", "content": "What is 1 + 1?"}]}] * 4)
    # beta=0 at init stops TRL re-downloading a reference model by name; we pass reference log-probs ourselves.
    args = GRPOConfig(output_dir=tempfile.mkdtemp(), use_cpu=True, report_to="none", beta=0.0, epsilon=0.2,
                      loss_type=loss_type, per_device_train_batch_size=4, num_generations=2, scale_rewards="group",
                      max_completion_length=8)
    trainer = GRPOTrainer(model=model, reward_funcs=lambda completions, **kw: [0.0] * len(completions), args=args,
                          train_dataset=data, processing_class=tok)
    trainer.beta = 0.04
    trainer.model.eval()
    device = trainer.accelerator.device

    torch.manual_seed(0)
    prompt_ids, prompt_mask = left_pad([[5, 6, 7, 8], [9, 10], [11, 12, 13], [14, 15, 16, 17]], tok.pad_token_id, 1)
    comp = torch.randint(3, 500, (4, 6))
    cmask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0]])
    comp = comp.masked_fill(cmask == 0, tok.pad_token_id)
    tensors = [t.to(device) for t in (prompt_ids, prompt_mask, comp, cmask)]
    advantages = torch.tensor([1.2, -0.7, 0.4, -1.1], device=device)
    with torch.no_grad():
        ref_logps = score_completions(ref.to(device), *tensors)
        logps = score_completions(trainer.model, *tensors)
        old = logps - 0.3 * torch.linspace(-1, 1, 6, device=device)  # a batch sampled by a slightly older policy
        ours, _ = grpo_loss(logps, old, advantages, tensors[3], ref_logps,
                            GRPOLossConfig(clip_eps=0.2, beta=0.04, loss_type=loss_type))
        theirs = trainer._compute_loss(trainer.model, {
            "prompt_ids": tensors[0], "prompt_mask": tensors[1], "completion_ids": tensors[2],
            "completion_mask": tensors[3], "advantages": advantages, "old_per_token_logps": old,
            "ref_per_token_logps": ref_logps, "num_items_in_batch": tensors[3].sum(),
        })
    assert abs(ours.item() - theirs.item()) < 1e-3


def test_exact_reward_is_strict():
    assert exact_reward("37 + 48 = 85. The answer is 85.", 85) == 1.0
    assert exact_reward("37 + 48 = 85. The answer is 86.", 85) == 0.0
    assert exact_reward("37 + 48 = 85.", 85) == 0.0  # no format, no reward
    assert extract_answer("The answer is 85. The answer is 3.") == 85  # the first claim counts
    assert ANSWER_RE.search("The answer is -4.")


def test_reinforce_sums_over_tokens_and_ignores_padding():
    logps = torch.tensor([[-1.0, -2.0, -9.0]], requires_grad=True)
    loss = reinforce_loss(logps, torch.tensor([[1, 1, 0]]), torch.tensor([1.0]))
    assert abs(loss.item() - 3.0) < 1e-6
    loss.backward()
    assert logps.grad.tolist() == [[-1.0, -1.0, 0.0]]


def test_estimators_zero_out_constant_rewards_except_vanilla():
    r = torch.ones(8)
    assert (estimator_weights(r, 4, "none") == 1).all()
    for name in ("batch", "group", "group_std"):
        assert estimator_weights(r, 4, name).abs().max() == 0


def test_gradient_noise_separates_signal_from_noise():
    torch.manual_seed(0)
    signal = torch.ones(8, 512)
    clean = gradient_noise(signal + 0.01 * torch.randn(8, 512))
    noisy = gradient_noise(signal + 10 * torch.randn(8, 512))
    assert clean["pairwise_cos"] > 0.99
    assert noisy["pairwise_cos"] < 0.1
    assert clean["snr"] > noisy["snr"]


@pytest.mark.network
@pytest.mark.parametrize("workers", [1, 2])
def test_rollout_engine_builds_a_consistent_batch(workers):
    tok = load_tokenizer()
    model = tiny_llama(vocab_size=len(tok))
    engine = RolloutEngine(model, tok, exact_reward, num_workers=workers, group_size=2, max_new_tokens=6)
    problems = sample_problems(4, seed=0)
    engine.sync(model, version=3)
    rollouts = engine.collect(problems)
    assert len(rollouts) == 8
    assert {r.policy_version for r in rollouts} == {3}
    assert len({r.worker_id for r in rollouts}) == workers
    # Whole groups stay together, so group means are over samples of one prompt.
    assert all(rollouts[i].problem is rollouts[i + 1].problem for i in range(0, 8, 2))
    b = engine.build_batch(rollouts)
    assert b["prompt_ids"].shape[0] == b["completion_ids"].shape[0] == 8
    assert b["completion_mask"].sum(-1).tolist() == [len(r.completion_ids) for r in rollouts]
    assert b["advantages"].shape == (8,)


def test_snr_of_pure_noise_sits_at_one_over_k_minus_one():
    torch.manual_seed(0)
    k = 12
    stats = gradient_noise(torch.randn(k, 20_000))
    assert stats["snr"] == pytest.approx(1 / (k - 1), rel=0.1)
    assert abs(stats["pairwise_cos"]) < 0.01


def test_worker_replica_only_changes_on_sync():
    from posttrain.rollout import GenerationWorker

    policy = tiny_llama()
    worker = GenerationWorker(policy, tok=None, worker_id=0, share_weights=False, max_new_tokens=4, temperature=1.0)
    with torch.no_grad():
        next(policy.parameters()).add_(1.0)
    p, w = next(policy.parameters()), next(worker.model.parameters())
    assert not torch.equal(p, w)  # stale until the trainer pushes weights
    worker.sync(policy, version=1)
    assert torch.equal(p, w) and worker.version == 1


def test_reference_copy_is_frozen():
    ref = frozen_copy(tiny_llama().train())
    assert not ref.training and not any(p.requires_grad for p in ref.parameters())


def test_random_policy_does_not_keep_the_pretrained_weights(tmp_path):
    tiny_llama().save_pretrained(tmp_path)
    loaded, fresh = load_policy(str(tmp_path)), random_policy(str(tmp_path))
    assert [p.shape for p in loaded.parameters()] == [p.shape for p in fresh.parameters()]
    assert not torch.equal(next(loaded.parameters()), next(fresh.parameters()))
    # The trap this guards against: re-running the init functions on loaded weights silently changes nothing.
    before = next(loaded.parameters()).clone()
    loaded.apply(loaded._init_weights)
    assert torch.equal(before, next(loaded.parameters()))
