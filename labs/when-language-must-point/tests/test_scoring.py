import numpy as np
import pytest

from pointlab.scoring import (boundary_iou, ce_logit_grad_norm, cross_entropy, expected_iou_reward,
                              generalised_iou, giou_ciou, match_f1, sequence_f1)

TRUE = (268, 584)
B = {583: 0.9, 584: 0.1}
C = {884: 0.9, 584: 0.1}


def test_cross_entropy_is_blind_to_distance():
    assert cross_entropy(B, 584) == pytest.approx(cross_entropy(C, 584)) == pytest.approx(2.302585, abs=1e-5)
    assert ce_logit_grad_norm(B, 584) == pytest.approx(ce_logit_grad_norm(C, 584))


def test_iou_reward_ranks_near_above_far():
    assert expected_iou_reward(B, TRUE) > 0.99
    assert expected_iou_reward(C, TRUE) < 0.6


def test_giou_vs_ciou():
    big, small = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    big[:80, :80] = True
    small[:4, :4] = True
    wrong_small = np.zeros_like(small)
    wrong_small[50:54, 90:94] = True
    g, c = giou_ciou([big, wrong_small], [big, small])
    assert g == pytest.approx(0.5)
    assert c > 0.99


def test_boundary_iou():
    m = np.zeros((40, 40), bool)
    m[5:35, 5:35] = True
    assert boundary_iou(m, m) == 1.0
    shifted = np.roll(m, 2, axis=1)
    assert boundary_iou(shifted, m) < 0.6 < 0.8 < float((shifted & m).sum() / (shifted | m).sum())


def test_generalised_iou_values():
    assert generalised_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert generalised_iou((0, 0, 10, 10), (20, 0, 30, 10)) == pytest.approx(-1 / 3)


def test_set_matching():
    cells = [(0, 0, 10, 10), (0, 20, 10, 30), (0, 40, 10, 50)]
    assert match_f1(cells[::-1], cells)["f1"] == 1.0                    # order does not matter
    assert match_f1(cells[1:], cells)["f1"] == pytest.approx(0.8)
    assert sequence_f1(cells[1:], cells)["f1"] == 0.0                   # one skip shifts everything after it
    assert match_f1([], [])["f1"] == 1.0                                # "no such region" is correct
    assert match_f1([(0, 0, 5, 5)], [])["f1"] == 0.0
