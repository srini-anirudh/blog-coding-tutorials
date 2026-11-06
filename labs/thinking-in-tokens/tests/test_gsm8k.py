from thinktok.gsm8k import extract_answer, gold_answer, last_boxed, normalise


def test_gold():
    assert gold_answer("She has 3 + 4 = 7 apples.\n#### 1,250") == "1250"


def test_extract():
    assert extract_answer("so the total is \\boxed{72}.") == "72"
    assert extract_answer("first 3, then \\boxed{\\$1,000.00}") == "1000"
    assert extract_answer("no box: 12 then 18") == "18"
    assert extract_answer("nothing here") is None
    assert last_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_normalise():
    assert normalise("5.50") == "5.5"
    assert normalise("-3") == "-3"
    assert normalise("40%") == "40"
