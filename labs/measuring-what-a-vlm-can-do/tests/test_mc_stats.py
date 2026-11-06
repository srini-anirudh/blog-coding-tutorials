"""Option parsing, extractors, CircularEval, intervals, MG / ML and Bradley-Terry fitting."""

import math

import pytest
import torch

from benchlab import mc
from benchlab import stats as S

Q = ("Which option describe the object relationship in the image correctly?\nOptions: A: The suitcase is on "
     "the book., B: The suitcase is beneath the cat., C: The suitcase is beneath the bed., D: The suitcase is beneath the book.")


def test_parse_mmstar():
    it = mc.parse_mmstar(Q, "A")
    assert it.options[0] == "The suitcase is on the book."
    assert it.options[3] == "The suitcase is beneath the book."
    assert it.gold_letter == "A"
    assert mc.format_prompt(it, hint=False) == Q
    with pytest.raises(ValueError):
        mc.parse_mmstar("no options here", "A")


def test_parse_mathvista_layout():
    q = ("Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
         "\nQuestion: What is the sum of 2002, 2003 and 2004?\nChoices:\n(A) 40.7\n(B) 74\n(C) 70.4\n(D) 70.0")
    it = mc.parse_mmstar(q, "C")
    assert it.question == "What is the sum of 2002, 2003 and 2004?"
    assert it.options == ["40.7", "74", "70.4", "70.0"] and it.gold_letter == "C"
    no_header = q.replace("Choices:\n", "")
    assert mc.parse_mmstar(no_header, "C") == it


def test_rotation_moves_gold_letter():
    it = mc.MCItem("q", ["w", "x", "y", "z"], 2)
    vs = mc.circular_variants(it)
    assert [v.gold_letter for v in vs] == ["C", "B", "A", "D"]
    assert all(v.options[v.answer] == "y" for v in vs)
    assert mc.circular_correct(["C", "B", "A", "D"], vs)
    assert not mc.circular_correct(["C", "B", "A", "A"], vs)


@pytest.mark.parametrize("reply,strict,perm", [
    ("B", "B", "B"), ("(B).", "B", "B"), ("D: Playing guitar.", None, "D"),
    ("Answer: C", None, "C"), ("The answer is B or C", None, None),
    ("A dog on a sofa", None, None), ("A: The skyline", None, "A"), ("E", None, None),
])
def test_extractors(reply, strict, perm):
    assert mc.extract_strict(reply, 4) == strict
    assert mc.extract_permissive(reply, 4) == perm


def test_content_extractor_falls_back_to_option_text():
    it = mc.MCItem("q", ["red bus", "blue car", "green tram", "a bike"], 1)
    assert mc.extract("It is a blue car.", it, "content") == "B"
    assert mc.extract("It is a blue car.", it, "permissive") is None
    assert mc.extract("red or blue", it, "content") is None


def test_standard_error_matches_post():
    assert 100 * S.standard_error(0.6, 1000) == pytest.approx(1.55, abs=0.01)
    assert 100 * S.standard_error(0.6, 218) == pytest.approx(3.3, abs=0.05)
    assert 100 * S.standard_error(0.6, 60) == pytest.approx(6.3, abs=0.05)


def test_bootstrap_intervals():
    x = torch.tensor([1.0] * 600 + [0.0] * 400)
    lo, hi = S.bootstrap_ci(x)
    assert lo < 0.6 < hi and hi - lo == pytest.approx(2 * 1.96 * S.standard_error(0.6, 1000), rel=0.15)
    d = S.paired_bootstrap(x, x)
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_mg_ml():
    out = S.multimodal_gain_leakage(0.5, 0.35, 0.30)
    assert out["MG"] == pytest.approx(0.15) and out["ML"] == pytest.approx(0.05)
    assert S.multimodal_gain_leakage(0.5, 0.2, 0.3)["ML"] == 0.0


def test_elo_and_bradley_terry_recovery():
    assert S.elo_win_prob(1400, 1000) == pytest.approx(10 / 11)
    true = torch.tensor([1150.0, 1000.0, 850.0])
    g = torch.Generator().manual_seed(1)
    i = torch.randint(0, 3, (20000,), generator=g)
    j = (i + torch.randint(1, 3, (20000,), generator=g)) % 3
    p = torch.sigmoid((true[i] - true[j]) * math.log(10) / 400)
    win = torch.rand(20000, generator=g) < p
    fit = S.fit_bradley_terry(torch.where(win, i, j), torch.where(win, j, i), 3)
    assert (fit - true).abs().max() < 15
