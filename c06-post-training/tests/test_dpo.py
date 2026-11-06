"""Stage 2 checks, including the debugging list: masking, padding, lengths, sum vs mean, reference gradients."""

import tempfile

import pytest
import torch
from datasets import Dataset

from posttrain.chat import load_tokenizer
from posttrain.dpo import collate_pairs, dpo_batch_loss, dpo_loss, synthetic_pairs, tokenize_pair
from posttrain.logprobs import per_token_logprobs, sequence_logprobs
from posttrain.models import frozen_copy, tiny_llama
from posttrain.tasks import sample_problems


def _batch():
    """Two right-padded rows: prompt of 3 tokens, completions of 4 and 2 tokens."""
    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11], [5, 6, 7, 12, 13, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]])
    comp = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 0]])
    return ids, attn, comp


def test_sequence_logprob_matches_explicit_loop():
    model = tiny_llama()
    ids, attn, comp = _batch()
    with torch.no_grad():
        got = sequence_logprobs(model, ids, attn, comp)
        logits = model(input_ids=ids, attention_mask=attn).logits
    log_p = logits.log_softmax(-1)
    for row in range(2):
        want = sum(log_p[row, t - 1, ids[row, t]] for t in range(ids.shape[1]) if comp[row, t])
        assert abs(got[row].item() - want.item()) < 1e-4


def test_prompt_tokens_do_not_count():
    model = tiny_llama()
    ids, attn, comp = _batch()
    with torch.no_grad():
        everything = sequence_logprobs(model, ids, attn, attn)
        completion = sequence_logprobs(model, ids, attn, comp)
    assert (everything < completion).all()  # log-probs are negative, so the extra prompt terms make it smaller


def test_padding_does_not_change_sequence_logprob():
    model = tiny_llama()
    ids, attn, comp = _batch()
    with torch.no_grad():
        padded = sequence_logprobs(model, ids, attn, comp)[1]
        alone = sequence_logprobs(model, ids[1:, :5], attn[1:, :5], comp[1:, :5])[0]
    assert abs(padded.item() - alone.item()) < 1e-4


def test_left_and_right_padding_agree():
    model = tiny_llama()
    with torch.no_grad():
        right = per_token_logprobs(model, torch.tensor([[5, 6, 7, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))[0, :2]
        left = per_token_logprobs(model, torch.tensor([[0, 0, 5, 6, 7]]), torch.tensor([[0, 0, 1, 1, 1]]))[0, 2:]
    torch.testing.assert_close(right, left, atol=1e-5, rtol=1e-5)


def test_sum_vs_mean_changes_the_margin():
    """Averaging removes the length term: a long reply no longer weighs more than a short one."""
    model = tiny_llama()
    ids, attn, comp = _batch()
    with torch.no_grad():
        total = sequence_logprobs(model, ids, attn, comp)
        mean = sequence_logprobs(model, ids, attn, comp, average=True)
    torch.testing.assert_close(mean, total / comp.sum(-1).float(), atol=1e-5, rtol=1e-5)
    assert not torch.isclose(total, mean).all()


def test_equal_policy_and_reference_give_log_two():
    model = tiny_llama()
    ref = frozen_copy(model)
    ids, attn, comp = _batch()
    batch = {"input_ids": ids.repeat(2, 1), "attention_mask": attn.repeat(2, 1), "completion_mask": comp.repeat(2, 1)}
    with torch.no_grad():
        loss, m = dpo_batch_loss(model, ref, batch, beta=0.1)
    assert abs(loss.item() - torch.log(torch.tensor(2.0)).item()) < 1e-5  # the margin starts at exactly 0
    assert m["margin"].abs().max() < 1e-5


def test_reference_gets_no_gradient():
    model = tiny_llama().train()
    ref = frozen_copy(model)
    ids, attn, comp = _batch()
    batch = {"input_ids": ids.repeat(2, 1), "attention_mask": attn.repeat(2, 1), "completion_mask": comp.repeat(2, 1)}
    batch["input_ids"][2:, 3] = 40  # make rejected differ from chosen so the gradient is not zero
    loss, _ = dpo_batch_loss(model, ref, batch)
    loss.backward()
    assert all(p.grad is None for p in ref.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert not ref.training


def test_dpo_loss_decreases_with_margin():
    zero = torch.zeros(1)
    low, _ = dpo_loss(zero, zero, zero, zero)
    high, _ = dpo_loss(zero + 5, zero, zero, zero)
    assert high < low


@pytest.mark.network
def test_matches_trl_dpo_trainer_on_the_same_batch():
    """Same tokens, same tensors, same loss as trl.DPOTrainer."""
    from trl import DPOConfig, DPOTrainer

    tok = load_tokenizer()
    model = tiny_llama(vocab_size=len(tok), seed=0).train()
    ref = tiny_llama(vocab_size=len(tok), seed=1).eval()  # different weights, so the log-ratios are not zero
    pairs = synthetic_pairs(sample_problems(4, seed=3))
    data = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])
    args = DPOConfig(output_dir=tempfile.mkdtemp(), use_cpu=True, report_to="none", beta=0.1,
                     per_device_train_batch_size=4)
    trainer = DPOTrainer(model=model, ref_model=ref, args=args, train_dataset=data, processing_class=tok)

    theirs_examples = [trainer.train_dataset[i] for i in range(4)]
    ours_examples = [tokenize_pair(tok, p) for p in pairs]
    for a, b in zip(theirs_examples, ours_examples):
        assert a["prompt_ids"] == b["prompt_ids"]
        assert a["chosen_ids"] == b["chosen_ids"]
        assert a["rejected_ids"] == b["rejected_ids"]

    theirs_batch = trainer.data_collator(theirs_examples)
    ours_batch = collate_pairs(ours_examples, tok.pad_token_id)
    for key in ours_batch:
        assert torch.equal(theirs_batch[key], ours_batch[key]), key

    trainer.model.eval()
    device = trainer.accelerator.device
    with torch.no_grad():
        theirs = trainer.compute_loss(trainer.model, {k: v.to(device) for k, v in theirs_batch.items()})
        ours, _ = dpo_batch_loss(trainer.model, trainer.ref_model, {k: v.to(device) for k, v in ours_batch.items()})
    assert abs(theirs.item() - ours.item()) < 1e-4
