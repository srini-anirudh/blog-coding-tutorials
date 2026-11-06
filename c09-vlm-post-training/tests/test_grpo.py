import torch

from vlmalign.grpo import Rollouts, grpo_loss, group_advantages, rollout_batch
from vlmalign.logprobs import token_logprobs, forward_logits
from vlmalign.tiny import tiny_pixels, tiny_prompt, tiny_vlm


def test_group_advantages_are_relative_within_each_image():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.2, 0.4, 0.6, 0.8]])
    a = group_advantages(r)
    torch.testing.assert_close(a.mean(1), torch.zeros(3), atol=1e-6, rtol=0)
    assert a[1].abs().max() == 0                      # all equal -> no learning signal
    assert torch.allclose(a[0].std(), torch.tensor(1.0), atol=1e-3)
    assert (a[2].argsort() == torch.arange(4)).all()  # order is preserved
    assert torch.allclose(group_advantages(r, scale_by_std=False)[0], torch.tensor([0.5, -0.5, -0.5, 0.5]))


def test_on_policy_grpo_gradient_is_advantage_times_score():
    logp = torch.tensor([[-1.0, -2.0], [-0.5, -0.1]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    adv = torch.tensor([2.0, -1.0])
    grpo_loss(logp, logp.detach(), adv, mask).backward()
    expected = -adv[:, None] * mask / mask.sum()
    torch.testing.assert_close(logp.grad, expected)


def test_clipping_stops_the_push_beyond_the_trust_region():
    old = torch.tensor([[-1.0]])
    logp = torch.tensor([[-1.0 + 0.5]], requires_grad=True)  # ratio e^0.5 = 1.65 > 1.2
    grpo_loss(logp, old, torch.tensor([1.0]), torch.ones(1, 1)).backward()
    assert logp.grad.item() == 0.0                           # positive advantage, already clipped
    logp2 = torch.tensor([[-1.0 + 0.5]], requires_grad=True)
    grpo_loss(logp2, old, torch.tensor([-1.0]), torch.ones(1, 1)).backward()
    assert logp2.grad.item() > 0                             # negative advantage still pulls down


def test_k3_kl_is_zero_at_reference_and_positive_elsewhere():
    logp = torch.tensor([[-1.0, -2.0]])
    mask = torch.ones(1, 2)
    zero_adv = torch.zeros(1)
    assert grpo_loss(logp, logp, zero_adv, mask, ref_logp=logp, beta=1.0).item() == 0
    assert grpo_loss(logp, logp, zero_adv, mask, ref_logp=logp - 0.3, beta=1.0).item() > 0


def test_rollout_batch_shares_one_image_per_group_and_masks_after_stop():
    eou, pad = 50, 0
    comp = torch.tensor([[7, eou, pad], [8, 9, eou], [10, 11, 12]])
    g = Rollouts(tiny_prompt(), tiny_pixels(1), comp, torch.tensor([2, 3, 3]), ["", "", ""])
    b = rollout_batch([g, g], pad_id=pad)
    assert b.image_index.tolist() == [0, 0, 0, 1, 1, 1]
    assert (b.labels != -100).sum(1).tolist() == [2, 3, 3, 2, 3, 3]
    logp, mask = token_logprobs(forward_logits(tiny_vlm(), b), b.labels)
    assert logp.shape == mask.shape == (6, b.input_ids.shape[1] - 1)
