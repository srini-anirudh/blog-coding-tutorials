import numpy as np
import pytest

from vlmalign import rewards as R


def test_exact_match_normalises():
    assert R.exact_match(" Red.", "red") == 1.0
    assert R.exact_match("three", "3") == 1.0
    assert R.exact_match("The circle", "circle") == 1.0
    assert R.exact_match("red circle", "red") == 0.0


def test_contains_match_is_gameable():
    listing = "1 2 3 4 5"
    assert R.exact_match(listing, "3") == 0.0
    assert R.contains_match(listing, "3") == 1.0  # the weak reward pays for a list


def test_box_iou_known_values():
    a = (0, 0, 10, 10)
    assert R.box_iou(a, a) == 1.0
    assert R.box_iou(a, (20, 20, 30, 30)) == 0.0
    # half overlap: inter 50, union 150
    assert R.box_iou(a, (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert R.box_iou(a, (5, 0, 15, 10)) == R.box_iou((5, 0, 15, 10), a)


def test_center_in_box_is_gameable_iou_is_not():
    gold = (100, 100, 150, 150)
    whole = (0, 0, 512, 512)
    assert R.center_in_box(whole, gold) == 1.0
    assert R.box_iou(whole, gold) < 0.01


def test_parse_box_and_box_reward():
    assert R.parse_box("[10, 20, 30, 40]") == (10, 20, 30, 40)
    assert R.parse_box("box: 30 20 10 40") is None  # x1 < x0
    assert R.parse_box("no box") is None
    assert R.box_reward("garbage", (0, 0, 1, 1)) == 0.0
    assert R.box_reward("0 0 10 10", (0, 0, 10, 10)) == 1.0


def test_mask_iou_and_dice():
    gold = np.zeros((8, 8), bool)
    gold[:4] = True
    pred = np.zeros((8, 8), bool)
    pred[2:6] = True
    iou, dice = R.mask_iou(pred, gold), R.mask_dice(pred, gold)
    assert iou == pytest.approx(16 / 48)
    assert dice == pytest.approx(2 * iou / (1 + iou))
    assert dice >= iou
    empty = np.zeros((4, 4), bool)
    assert R.mask_iou(empty, empty) == 1.0 and R.mask_dice(empty, empty) == 1.0


def test_levenshtein_cer_anls():
    assert R.levenshtein("kitten", "sitting") == 3
    assert R.levenshtein("", "abc") == 3
    assert R.cer("CAT", "CAT") == 0.0
    assert R.cer("CUT", "CAT") == pytest.approx(1 / 3)
    assert R.anls("CAT", "cat") == 1.0                 # case-insensitive
    assert R.anls("CUT", "CAT") == pytest.approx(2 / 3)
    assert R.anls("DOG", "CAT") == 0.0                 # similarity below 0.5 is cut to zero
    assert R.anls("CAT", ["DOG", "CAT"]) == 1.0        # best over gold variants


def test_field_accuracy():
    gold = {"name": "Ada", "total": "42"}
    assert R.field_accuracy('{"name": "Ada", "total": "42"}', gold) == 1.0
    assert R.field_accuracy('{"name": "Ada", "total": "41", "x": 1}', gold) == 0.5
    assert R.field_accuracy('{"name": "Ada",', gold) == 0.0
    assert R.field_accuracy('["Ada"]', gold) == 0.0


def test_caption_recall_rewards_hallucination_f1_does_not():
    vocab = ["red", "green", "blue", "yellow", "purple"]
    truth = {"red", "blue"}
    honest = "a red circle and a blue square"
    spray = "red green blue yellow purple"
    assert R.caption_recall(spray, truth, vocab) == 1.0
    assert R.caption_recall(honest, truth, vocab) == 1.0
    assert R.caption_f1(honest, truth, vocab) == 1.0
    assert R.caption_f1(spray, truth, vocab) == pytest.approx(2 * 0.4 * 1 / 1.4)
