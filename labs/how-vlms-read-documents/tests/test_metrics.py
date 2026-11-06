import pytest

from docread.metrics import (anls, anls_single, check_absent, check_no_repetition, check_order, check_present,
                             levenshtein, longest_repeat_run, normalized_edit_distance, omnidocbench_overall)


def test_levenshtein_strings_and_lists():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein(["a", "b", "c"], ["a", "c"]) == 1
    assert normalized_edit_distance("", "") == 0.0


def test_anls_matches_the_post():
    assert anls_single("1,580.00", ["1,500.00"]) == pytest.approx(0.875)
    assert anls_single("15", ["1,500.00"]) == 0.0  # NL 0.75 is past the 0.5 threshold
    assert anls_single("northwind", ["Northwind Traders", "Northwind"]) == 1.0  # case-insensitive, best gold
    assert anls(["a", "b"], [["a"], ["c"]]) == 0.5


def test_omnidocbench_overall_paddleocr_vl():
    assert omnidocbench_overall(0.035, 90.89, 91.22) == pytest.approx(92.86, abs=0.02)


def test_unit_checks():
    out = "Title\nBody text here.\nFooter"
    assert check_present(out, "Body  text here.")
    assert check_absent(out, "Page 7")
    assert check_order(out, "Title", "Footer") and not check_order(out, "Footer", "Title")
    assert longest_repeat_run("a\nb\nb\nb\nc") == 3
    assert not check_no_repetition("x\n" * 5)
