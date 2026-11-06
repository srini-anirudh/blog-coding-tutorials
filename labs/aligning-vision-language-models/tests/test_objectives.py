import math

import torch

from groundpref.objectives import (anchor_loss, changed_mask, conditional_loss, dense_weights, dpo_loss,
                                   margin_split)


def test_dpo_loss_is_log2_at_the_reference():
    x = torch.randn(8)
    loss, margin = dpo_loss(x, x - 1, x, x - 1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    assert torch.allclose(margin, torch.zeros(8))


def test_margin_split_is_an_identity():
    torch.manual_seed(0)
    pol = {k: torch.randn(16) for k in ("w_img", "w_null", "l_img", "l_null")}
    ref = {k: torch.randn(16) for k in pol}
    s = margin_split(pol, ref, beta=0.3)
    assert math.isclose(s["beta_d_text"] + s["beta_d_image"], s["beta_margin"], rel_tol=1e-5, abs_tol=1e-6)


def test_conditional_term_ignores_any_text_only_shift():
    """Add the same text-only log-prob to both sides of the image contrast: the loss does not move."""
    torch.manual_seed(1)
    pw, pb, rw, rb = (torch.randn(8) for _ in range(4))
    shift = 5 * torch.randn(8)
    a = conditional_loss(pw, pb, rw, rb)
    b = conditional_loss(pw + shift, pb + shift, rw, rb)
    assert torch.allclose(a, b)


def test_anchor_penalises_pushing_the_chosen_response_down():
    rw = torch.zeros(4)
    assert anchor_loss(rw - 5, rw) > anchor_loss(rw + 5, rw)


def test_changed_mask_and_dense_weights_mark_the_edit():
    a, b = [1, 2, 3, 4, 5], [1, 2, 9, 4, 5]
    ma, mb = changed_mask(a, b)
    assert ma == [False, False, True, False, False] and mb == ma
    w = dense_weights(torch.tensor([a]), [ma], gamma=5.0)
    assert w.tolist() == [[1.0, 5.0, 1.0, 1.0]]
