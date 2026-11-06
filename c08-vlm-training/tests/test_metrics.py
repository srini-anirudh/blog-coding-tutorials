"""Every metric checked against values worked out by hand."""

import math

import pytest
import torch

from vlmtrain import metrics as M


def test_normalize_answer():
    assert M.normalize_answer("  The Red. ") == "red"
    assert M.normalize_answer("Two") == "2"
    assert M.normalize_answer("37.50") == "37.50"  # decimal point survives
    assert M.normalize_answer("a blue square!") == "blue square"


def test_exact_match():
    assert M.exact_match("Yes.", "yes") == 1.0
    assert M.exact_match("three", "3") == 1.0
    assert M.exact_match("4", ["3", "four"]) == 1.0
    assert M.exact_match("blue", "red") == 0.0


def test_levenshtein_known_values():
    assert M.levenshtein("kitten", "sitting") == 3
    assert M.levenshtein("", "abc") == 3
    assert M.levenshtein("abc", "abc") == 0
    assert M.levenshtein(["a", "b"], ["b"]) == 1


def test_cer_wer():
    assert M.cer("helo", "hello") == pytest.approx(1 / 5)
    assert M.wer("green light", "green night") == pytest.approx(1 / 2)
    assert M.wer("a b c d", "a b") == pytest.approx(1.0)  # two insertions over two words
    assert M.cer("", "") == 0.0


def test_anls():
    # NL = 1/6 for "Elena" vs "Elena." -> 1 - 1/6
    assert M.anls("Elena", "Elena.") == pytest.approx(1 - 1 / 6)
    assert M.anls("ELENA ROSSI", "elena rossi") == 1.0
    # distance 4 over length 5 -> NL = 0.8 >= 0.5 -> 0, not 0.2
    assert M.anls("37.50", "12.00") == 0.0
    assert M.anls("4821", ["9999", "4822"]) == pytest.approx(0.75)


def test_relaxed_accuracy():
    assert M.relaxed_accuracy("7", "7") == 1.0
    assert M.relaxed_accuracy("about 10.4", "10") == 1.0  # 4% off
    assert M.relaxed_accuracy("10.6", "10") == 0.0        # 6% off
    assert M.relaxed_accuracy("owl", "owl") == 1.0         # non-numeric falls back to exact match
    assert M.relaxed_accuracy("no number", "3") == 0.0
    assert M.parse_number("1,250%") == 1250.0


def test_box_iou():
    assert M.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert M.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert M.box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert M.point_in_box((5, 5), (0, 0, 10, 10)) == 1.0
    assert M.point_in_box((11, 5), (0, 0, 10, 10)) == 0.0
    assert not math.isnan(M.box_iou((0, 0, 0, 0), (0, 0, 0, 0)))


@pytest.mark.parametrize("fmt,limit", [("tokens", 24), ("text", 100)])
def test_evaluate_gives_each_grounding_format_its_own_limit(fmt, limit):
    from vlmtrain.c07_adapter import build_c07_tiny
    from vlmtrain.evaluate import evaluate

    _, model, proc = build_c07_tiny()
    seen = []
    real_generate = model.generate

    def spy(*args, **kwargs):
        seen.append(kwargs["max_new_tokens"])
        kwargs["max_new_tokens"] = 1
        return real_generate(*args, **kwargs)

    model.generate = spy
    evaluate(model, proc, torch.device("cpu"), ["vqa", "grounding"], n=1, grounding_format=fmt)
    assert seen == [24, limit]
