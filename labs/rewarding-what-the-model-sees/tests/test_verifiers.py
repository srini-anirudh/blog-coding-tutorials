import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrl import verifiers as V  # noqa: E402

TARGET = (0.25, 0.30, 0.75, 0.70)   # 20% of the unit image
WHOLE = (0.0, 0.0, 1.0, 1.0)


def test_whole_image_box_earns_the_target_share():
    assert V.box_iou(WHOLE, TARGET) == pytest.approx(0.2)


def test_giou_equals_iou_for_any_containing_box():
    for box in [WHOLE, (0.2, 0.2, 0.8, 0.8), TARGET]:
        assert V.giou(box, TARGET) == pytest.approx(V.box_iou(box, TARGET))


def test_giou_ranks_misses():
    near, far = (0.76, 0.30, 1.0, 0.70), (0.0, 0.0, 0.24, 0.10)
    assert V.box_iou(near, TARGET) == V.box_iou(far, TARGET) == 0
    assert V.giou(near, TARGET) > V.giou(far, TARGET)


def test_coverage_gives_whole_image_full_credit():
    assert V.coverage(WHOLE, TARGET) == pytest.approx(1.0)


def test_threshold_passes_containing_box_only_up_to_twice_the_area():
    assert V.iou_at((0.25, 0.12, 0.75, 0.88), TARGET)          # 1.9x
    assert not V.iou_at((0.25, 0.08, 0.75, 0.92), TARGET)      # 2.1x


def test_dice_is_at_least_iou_and_matches_the_identity():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.random((8, 8)) > 0.5, rng.random((8, 8)) > 0.5
        iou, dice = V.mask_iou(a, b), V.mask_dice(a, b)
        assert dice >= iou - 1e-12
        assert dice == pytest.approx(2 * iou / (1 + iou))


def test_relaxed_accuracy_band_and_strict_parse():
    assert V.relaxed_accuracy("57.8", "58") == 1
    assert V.relaxed_accuracy("56", "58") == 1          # a misreading inside 5%
    assert V.relaxed_accuracy("55", "58") == 0
    assert V.relaxed_accuracy("between 56 and 60", "58", strict=False) == 1
    assert V.relaxed_accuracy("between 56 and 60", "58") == 0
    assert V.relaxed_accuracy("Paris", "paris") == 1


def test_parsers():
    assert V.lenient_choice("B or C", "C") == 1
    assert V.strict_choice("B or C", "C") == 0
    assert V.strict_choice("I think so. <answer>C</answer>", "C") == 1
    assert V.strict_choice("<answer>B</answer> <answer>C</answer>", "C") == 0


def test_points():
    assert V.point_in_box((0.5, 0.5), TARGET) == 1
    assert V.point_distance((0.0, 0.0), (0.3, 0.4), d=1.0) == pytest.approx(0.5)
    assert V.point_distance((0.0, 0.0), (3, 4), d=1.0) == 0


def test_matching_is_optimal_not_greedy():
    golds = [(0, 0, 2, 2), (1, 0, 3, 2)]
    preds = [(1, 0, 3, 2), (0, 0, 2, 2)]
    pairs = V.match(preds, golds)
    assert sorted((p, g) for p, g, _ in pairs) == [(0, 1), (1, 0)]
    assert all(iou == pytest.approx(1) for *_, iou in pairs)


def test_recall_precision():
    golds = [(0, 0, 1, 1), (2, 2, 3, 3)]
    r, p = V.recall_precision([(0, 0, 1, 1), (5, 5, 6, 6), (7, 7, 8, 8)], golds)
    assert r == pytest.approx(0.5)
    assert p == pytest.approx(1 / 3)


def test_listing_every_category_is_free_until_length_aware():
    gold = {"dog": [(0, 0, 1, 1)]}
    every = {c: [(0, 0, 1, 1)] for c in ["dog", "cat", "car", "bus"]}
    assert V.category_score(every, gold) == 1.0
    assert V.length_aware(1.0, 4, 1) == pytest.approx(0.25)


def test_json_gate_and_f1():
    schema = {"vendor", "total", "date"}
    gold = {"vendor": "Acme", "total": "4.20"}
    assert V.json_field_f1('{"vendor": "acme", "total": "4.20"}', gold, schema) == 1
    assert V.json_field_f1("not json", gold, schema) == 0
    assert V.json_field_f1('{"vendor": "Acme", "colour": "red"}', gold, schema) == 0   # unknown key
    assert V.json_field_f1("{}", gold, schema) == 0
    guess = '{"vendor": "Acme", "total": "4.20", "date": "1999"}'
    assert V.json_field_recall(guess, gold, schema) == 1
    assert V.json_field_f1(guess, gold, schema) == pytest.approx(0.8)


def test_anls_threshold():
    assert V.anls("TOTAL 41.20", "TOTAL 41.20") == 1
    assert 0.5 < V.anls("T0TAL 41.20", "TOTAL 41.20") < 1
    assert V.anls("completely different", "TOTAL 41.20") == 0
