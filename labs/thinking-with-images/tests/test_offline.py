"""Offline tests: arithmetic, scenes, preprocessing, answer parsing, and the probe and search logic on a stub reader."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lookagain import budget  # noqa: E402
from lookagain.answers import FIND, VERIFY, closing_time, majority, times_in  # noqa: E402
from lookagain.probes import (answer_region_view, ask_views, blind_prior, caption_then_reason, oracle_swap,  # noqa: E402
                              pass_and_vote, samples_vs_views, score)
from lookagain.scene import CLOSE_TIMES, GRID, TEXT_H, H, W, cell_box, make_scene  # noqa: E402
from lookagain.search import best_first, children, exhaustive  # noqa: E402
from lookagain.stub import StubVLM  # noqa: E402
from lookagain.views import View, cell_view, detail_retained, full_view, random_view, render  # noqa: E402
from lookagain.vlm import n_tokens, patchify  # noqa: E402


# ---- the post's arithmetic ---------------------------------------------------------------------------------------
def test_post_numbers():
    assert budget.Encoding(W, H, W, H).patches == 62_208
    small = budget.Encoding(W, H, 448, 336)
    assert small.grid == (32, 24) and small.tokens == 768
    assert small.text_height(32) == pytest.approx(3.56, abs=0.01)
    assert budget.min_width_for_text(32, 12, W) == pytest.approx(1512)
    assert budget.Encoding(W, H, 1512, 1134).tokens == 8_748
    crop = budget.Encoding(1008, 756, 448, 336)
    assert crop.tokens == 768 and crop.text_height(32) == pytest.approx(14.2, abs=0.05)
    assert budget.image_share(768, 5000) == pytest.approx(0.133, abs=0.001)
    assert budget.Encoding(W, H, 448, 336, merge=2).tokens == 192


def test_attention_cost_matches_sum():
    nv, t = 768, 300
    assert budget.attention_cost(nv, t) == sum(nv + i for i in range(1, t + 1))


# ---- scenes and views --------------------------------------------------------------------------------------------
def test_scene_layout():
    s = make_scene(0)
    assert s.image.size == (W, H)
    assert s.close_time in CLOSE_TIMES and s.bus_time != s.close_time
    assert cell_view(*s.plaque_cell).contains(s.plaque_box)
    assert s.plaque_cell == (GRID - 1, 0)
    assert s.close_time in s.facts and "closes" not in s.facts  # facts, not the answer
    s2 = make_scene(0)
    assert np.array_equal(np.asarray(s.image), np.asarray(s2.image))  # deterministic


def test_random_cell_scene_and_bus_elsewhere():
    for seed in range(10):
        s = make_scene(seed, cell=None)
        assert cell_view(*s.plaque_cell).contains(s.plaque_box)
        assert not cell_view(*s.plaque_cell).contains(s.bus_box)


def test_digit_height_is_32px():
    """Render the plaque digits alone and measure the ink height."""
    from PIL import Image, ImageDraw
    from lookagain.scene import font_for_digit_height
    img = Image.new("L", (400, 100), 0)
    ImageDraw.Draw(img).text((10, 10), "0123456789", font=font_for_digit_height(TEXT_H), fill=255)
    rows = np.where(np.asarray(img).max(1) > 0)[0]
    assert abs((rows[-1] - rows[0] + 1) - TEXT_H) <= 1


def test_view_text_px_and_render():
    v = full_view(448)
    assert (v.enc_w, v.enc_h) == (448, 336) and v.text_px == pytest.approx(3.56, abs=0.01)
    s = make_scene(1)
    img = render(s, answer_region_view(s))
    assert img.size == (448, 336) and img.info["view"].text_px == pytest.approx(14.2, abs=0.05)
    rv = random_view(random.Random(0))
    assert rv.enc_w % 28 == 0 and rv.enc_h % 28 == 0


def test_resize_loses_the_text():
    """Low-pass filter: the plaque's letter edges are mostly gone at 448 px and come back in the crop."""
    s = make_scene(2)
    full, crop = detail_retained(s, full_view(448)), detail_retained(s, answer_region_view(s))
    assert full < 0.3 < 0.6 < crop
    assert detail_retained(s, full_view(896)) > full


# ---- preprocessing -------------------------------------------------------------------------------------------------
def test_patchify_shapes_and_order():
    from PIL import Image
    img = Image.new("RGB", (56, 28))
    rows, grid = patchify(img)
    assert grid.tolist() == [1, 2, 4] and rows.shape == (8, 3 * 2 * 14 * 14)
    assert n_tokens(img) == 2
    # Mark the top-left 14 x 14 patch; it must be row 0. The patch to its right (same 2x2 group) is row 1.
    a = np.zeros((28, 56, 3), dtype=np.uint8)
    a[:14, 14:28] = 255
    rows, _ = patchify(Image.fromarray(a))
    bright = rows.mean(1)
    assert bright.argmax().item() == 1


# ---- answers -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,want", [
    ("17:30", "17:30"), ("The bakery closes at 5:30 PM.", "17:30"), ("OPEN 9:00-17:30", "17:30"),
    ("It is open from 8:00 to 16:00", "16:00"), ("I cannot tell.", None), ("Answer: 18:00", "18:00"),
])
def test_closing_time(text, want):
    assert closing_time(text) == want


def test_times_and_majority():
    assert times_in("9.00 and 21:00") == ["9:00", "21:00"]
    assert majority(["17:30", None, "18:00", "17:30"]) == "17:30"
    assert majority([None, None]) is None


# ---- probes and search on the stub reader ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scenes():
    return [make_scene(100 + i) for i in range(12)]


def test_lost_detail_pattern(scenes):
    vlm = StubVLM(legible_px=9)
    prior = blind_prior(vlm)
    low = score(scenes, ask_views(vlm, scenes, full_view(448)), prior)
    crop = score(scenes, ask_views(vlm, scenes, [answer_region_view(s) for s in scenes]), prior)
    blind = score(scenes, ask_views(vlm, scenes, None), prior)
    assert crop["acc"] == 1.0
    assert low["said_prior"] == 1.0 and blind["said_prior"] == 1.0  # the prior, with or without a smudge
    assert score(scenes, oracle_swap(vlm, scenes))["acc"] == 1.0


def test_caption_then_reason(scenes):
    vlm = StubVLM(legible_px=9)
    preds, has = caption_then_reason(vlm, scenes, full_view(2016))
    assert all(has) and score(scenes, preds)["acc"] == 1.0
    _, has = caption_then_reason(vlm, scenes, full_view(448))
    assert not any(has)


def test_samples_share_the_blind_spot(scenes):
    vlm = StubVLM(legible_px=9)
    samples, views, weights = samples_vs_views(vlm, scenes, k=8)
    s, v = pass_and_vote(scenes, samples, [8]), pass_and_vote(scenes, views, [8], weights)
    assert all(len(set(a)) == 1 for a in samples)  # every sample sees the same smudge
    assert v[8]["pass"] >= s[8]["pass"]
    assert v[8]["wvote"] >= v[8]["vote"]  # weighting by the verifier cannot hurt a reader whose verifier is right


def test_children_tile_the_parent():
    kids = children(full_view(448))
    assert len(kids) == 4
    assert sum((k.box[2] - k.box[0]) * (k.box[3] - k.box[1]) for k in kids) == W * H
    grand = {k2.box for k in kids for k2 in children(k)}
    assert grand == {cell_box(c, r) for c in range(GRID) for r in range(GRID)}


def test_search_finds_the_plaque_and_verification_rejects_the_bus():
    rand = [make_scene(200 + i, cell=None) for i in range(10)]
    plain = StubVLM(legible_px=9)
    for s in rand:
        tr = best_first(plain, s, full_view(448))
        assert tr.answer == s.close_time
        assert tr.visual_tokens < 16 * 192 * 2
    # A reader whose region scores prefer the bus sign: verification must back out of that branch.
    biased = StubVLM(legible_px=9)
    orig = biased.yes_prob

    def bus_first(items):
        out = orig(items)
        return [0.95 if p == FIND and ims[0].info["view"].contains(ims[0].info["scene"].bus_box) else o
                for (ims, p), o in zip(items, out)]

    biased.yes_prob = bus_first
    s = rand[0]
    no_verify = best_first(biased, s, full_view(448), verify=False)
    with_verify = best_first(biased, s, full_view(448), verify=True)
    assert no_verify.answer == s.bus_time
    assert with_verify.answer == s.close_time and with_verify.rejected == 1


def test_exhaustive_reads_all_cells():
    s = make_scene(300, cell=None)
    tr = exhaustive(StubVLM(), s, [cell_view(c, r) for r in range(GRID) for c in range(GRID)])
    assert tr.answer == s.close_time and tr.calls == 32 and tr.visual_tokens == 32 * 192


def test_stub_verify_prompt_distinguishes_bus():
    s = make_scene(5)
    vlm = StubVLM()
    bus_cell = next(cell_view(c, r) for r in range(GRID) for c in range(GRID)
                    if cell_view(c, r).contains(s.bus_box))
    p_bus, p_plaque = vlm.yes_prob([([render(s, bus_cell)], VERIFY), ([render(s, answer_region_view(s))], VERIFY)])
    assert p_bus < 0.5 < p_plaque


def test_drift_binning():
    from lookagain.drift import binned
    assert binned(torch.arange(8.0), 4) == [1.5, 5.5]


def test_distractor_off_changes_only_the_bus_sign():
    a, b = make_scene(7), make_scene(7, distractor=False)
    diff = np.abs(np.asarray(a.image, int) - np.asarray(b.image, int)).sum(2) > 0
    ys, xs = np.where(diff)
    x0, y0, x1, y1 = a.bus_box
    assert xs.min() >= x0 and xs.max() <= x1 and ys.min() >= y0 and ys.max() <= y1
    assert a.plaque_box == b.plaque_box and "BUS" not in b.facts
    assert not full_view(448).contains(b.bus_box)
    assert StubVLM().ask([render(b, full_view(2016))], "q") == b.open_time + "-" + b.close_time
