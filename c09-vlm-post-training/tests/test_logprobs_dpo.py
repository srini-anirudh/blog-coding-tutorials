import math

import torch

from vlmalign.dpo import dpo_loss
from vlmalign.logprobs import IGNORE, build_batch, forward_logits, sequence_logprobs
from vlmalign.tiny import tiny_pixels, tiny_prompt, tiny_vlm


def _batch():
    prompts = [tiny_prompt(3, 0), tiny_prompt(3, 0), tiny_prompt(5, 1), tiny_prompt(5, 1)]
    answers = [torch.tensor([7, 8, 9]), torch.tensor([7, 10]), torch.tensor([11, 12]), torch.tensor([13, 14, 15, 16])]
    return build_batch(prompts, answers, [0, 0, 1, 1], tiny_pixels(2), pad_id=0)


def test_batch_shapes_and_label_mask():
    b = _batch()
    assert b.input_ids.shape == b.labels.shape == b.attention_mask.shape == (4, 14)
    assert b.pixel_values.shape == (2, 1, 3, 32, 32)
    # labels exist only on answer tokens
    assert (b.labels != IGNORE).sum(1).tolist() == [3, 2, 2, 4]
    assert b.labels[0, : len(tiny_prompt(3, 0))].eq(IGNORE).all()


def test_encode_once_equals_naive_path():
    model, b = tiny_vlm(), _batch()
    torch.testing.assert_close(forward_logits(model, b, True), forward_logits(model, b, False))


def test_sequence_logprob_matches_hf_loss():
    """Our summed log-prob equals -(HF mean cross-entropy) * (number of answer tokens)."""
    model, b = tiny_vlm(), _batch()
    ours, n = sequence_logprobs(model, b)
    for i in range(4):
        out = model(input_ids=b.input_ids[i: i + 1], attention_mask=b.attention_mask[i: i + 1],
                    pixel_values=b.pixel_values[b.image_index[i: i + 1]], labels=b.labels[i: i + 1])
        torch.testing.assert_close(ours[i], -out.loss * n[i], rtol=1e-5, atol=1e-5)


def test_padding_does_not_change_a_row():
    model, b = tiny_vlm(), _batch()
    batched, _ = sequence_logprobs(model, b)
    alone = build_batch([tiny_prompt(3, 0)], [torch.tensor([7, 10])], [0], b.pixel_values[:1], pad_id=0)
    single, _ = sequence_logprobs(model, alone)
    torch.testing.assert_close(batched[1], single[0], rtol=1e-5, atol=1e-5)


def test_dpo_loss_is_log2_when_policy_equals_reference():
    lp = torch.tensor([-3.0, -5.0])
    out = dpo_loss(lp, lp - 1, lp, lp - 1, beta=0.1)
    assert math.isclose(out["loss"].item(), math.log(2), rel_tol=1e-6)
    assert out["margin"].abs().max() == 0


def test_dpo_gradient_raises_chosen_and_lowers_rejected():
    c = torch.tensor([-4.0], requires_grad=True)
    r = torch.tensor([-4.0], requires_grad=True)
    dpo_loss(c, r, torch.tensor([-4.0]), torch.tensor([-4.0]), beta=0.5)["loss"].backward()
    # d loss / d c = -beta * sigmoid(-margin) = -0.25 at margin 0
    assert c.grad.item() == -0.25 and r.grad.item() == 0.25


def test_one_dpo_step_increases_the_margin():
    model, b = tiny_vlm(), _batch()
    with torch.no_grad():
        ref, _ = sequence_logprobs(model, b)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    margins = []
    for _ in range(2):
        lp, _ = sequence_logprobs(model, b)
        out = dpo_loss(lp[0::2], lp[1::2], ref[0::2], ref[1::2], beta=1.0)
        margins.append(out["margin"].mean().item())
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    assert margins[0] == 0 and margins[1] > 0
