import pytest

from pointlab.coords import (BY_NAME, CONVENTIONS, brute_force_worst_iou, digit_error_shift, digit_split_count,
                             quantise, smart_resize, worst_case_iou)

BOX = (0.268, 0.372, 0.584, 0.811)
POST_TABLE = {
    "Shikra": "[0.268,0.372,0.584,0.811]",
    "Qwen-VL": "<ref>the dog</ref><box>(268,372),(584,811)</box>",
    "CogVLM": "[[268,372,584,811]]",
    "MiniGPT-v2": "{<27><37><58><81>}",
    "Florence-2": "<loc_268><loc_372><loc_584><loc_811>",
    "PaliGemma": "<loc0380><loc0274><loc0830><loc0598> dog",
    "Qwen2.5-VL": '{"bbox_2d": [233, 240, 507, 522], "label": "dog"}',
    "Molmo (point)": '<point x="42.6" y="59.2" alt="dog">dog</point>',
}


@pytest.mark.parametrize("name", list(POST_TABLE))
def test_encodings_match_the_post(name):
    assert BY_NAME[name].encode(BOX) == POST_TABLE[name]


@pytest.mark.parametrize("conv", CONVENTIONS, ids=lambda c: c.name)
def test_round_trip_within_one_bin(conv):
    back = conv.parse(conv.encode(BOX))
    if conv.name.startswith("Molmo"):
        assert back[0] == pytest.approx(0.426, abs=1e-3) and back[1] == pytest.approx(0.592, abs=1e-3)
    else:
        assert max(abs(a - b) for a, b in zip(back, BOX)) <= 0.005 + 1e-9


def test_location_tokens_are_short():
    florence = BY_NAME["Florence-2"]
    assert digit_split_count(florence.encode(BOX), florence.special_tokens) == 4
    assert digit_split_count("[[268,372,584,811]]") == 19   # every digit and mark is a token


@pytest.mark.parametrize("s,delta", [(24, 3.84), (100, 10.0), (30, 2.0), (40, 19.2)])
def test_worst_case_formula_matches_brute_force(s, delta):
    bf, errs = brute_force_worst_iou(s, delta)
    assert bf == pytest.approx(max(0, (s - delta) / s) ** 2, abs=1e-9)
    assert errs == (delta / 2, delta / 2, -delta / 2, -delta / 2)   # all four edges inward


def test_post_quantisation_numbers():
    assert worst_case_iou(24, 3840, 1000) == pytest.approx(0.7056, abs=1e-4)   # "about 0.71"
    assert worst_case_iou(24, 3840, 100) == 0.0                                # bin wider than the icon
    assert 2000 / 32 == 62.5                                                   # Kosmos-2 cell on a page scan
    assert quantise(5.0, 100, 10) == 5.0 and quantise(9.99, 100, 10) == 5.0


def test_digit_errors():
    assert digit_error_shift(584, 884) == pytest.approx(0.30)
    assert digit_error_shift(584, 589) == pytest.approx(0.005)


def test_smart_resize():
    assert smart_resize(644, 868) == (644, 868)                     # already multiples of 28
    assert smart_resize(2160, 3840, max_pixels=1280 * 28 * 28) == (728, 1316)
    h, w = smart_resize(2160, 3840, factor=32)
    assert h % 32 == 0 and w % 32 == 0
