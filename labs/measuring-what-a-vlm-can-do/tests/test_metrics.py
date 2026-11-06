"""The metric code against the worked examples in the post."""

import pytest
import torch

from benchlab import metrics as M

HUMANS = ["red"] * 4 + ["pink"] * 3 + ["maroon"] * 2 + ["orange"]


@pytest.mark.parametrize("ans,expected", [("red", 1.0), ("pink", 0.9), ("maroon", 0.6), ("orange", 0.3), ("blue", 0.0)])
def test_vqa_accuracy_figure_2(ans, expected):
    assert M.vqa_accuracy(ans, HUMANS) == pytest.approx(expected)


def test_vqa_normalize():
    assert M.vqa_normalize("Two.") == "2"
    assert M.vqa_normalize("The Bus!") == "bus"
    assert M.vqa_normalize("dont") == "don't"
    assert M.vqa_normalize("3.5") == "3.5"
    assert M.vqa_accuracy("there are 2 people", ["2"] * 10) == 0.0


def test_anls_examples():
    assert M.anls("SPRlNGFIELD", "SPRINGFIELD") == pytest.approx(10 / 11)
    assert M.anls("7,250", "1,250") == pytest.approx(0.8)
    assert M.anls("SHELBYVILLE", "SPRINGFIELD") == 0.0  # beyond the 0.5 threshold


def test_relaxed_accuracy_is_lenient_on_years():
    assert M.relaxed_accuracy("2020", "2019") == 1.0
    assert M.relaxed_accuracy("104", "100") == 1.0
    assert M.relaxed_accuracy("106", "100") == 0.0


def test_same_shift_large_and_small_box():
    big, small = (100, 100, 400, 400), (100, 100, 130, 130)
    assert M.box_iou((118, 100, 418, 400), big) > 0.5
    assert M.box_iou((118, 100, 148, 130), small) == pytest.approx(0.25)
    assert M.acc_at_iou([(118, 100, 418, 400), (118, 100, 148, 130)], [big, small]) == 0.5


def test_giou_ciou_large_and_small():
    gl = torch.zeros(64, 64, dtype=torch.bool); gl[:40, :25] = True        # 1000 px
    pl = gl.clone(); pl[36:40] = False                                       # IoU 0.9
    gs = torch.zeros(64, 64, dtype=torch.bool); gs[50:55, 50:54] = True      # 20 px
    ps = torch.zeros_like(gs); ps[50, 50:54] = True                          # IoU 0.2
    gi, ci = M.giou_ciou([pl, ps], [gl, gs])
    assert gi == pytest.approx(0.55)
    assert ci == pytest.approx((900 + 4) / (1000 + 20))


def test_pope_yes_ratio_and_parser():
    gold = ["yes"] * 5 + ["no"] * 5
    s = M.pope_stats(["yes"] * 8 + ["no"] * 2, gold)
    assert s["yes_ratio"] == 0.8 and s["recall"] == 1.0 and s["precision"] == pytest.approx(5 / 8)
    assert M.yes_no("No, there is not.") == "no"
    assert M.yes_no("Yes.") == "yes"
    assert M.yes_no("A cat.") is None
    assert M.pope_stats([None, "yes"], ["yes", "yes"])["accuracy"] == 0.5  # unparsed counts as 'no'


def test_chair():
    out = M.chair([{"dog", "cat"}, {"car"}], [{"dog"}, {"car"}])
    assert out == {"chair_i": pytest.approx(1 / 3), "chair_s": 0.5}
