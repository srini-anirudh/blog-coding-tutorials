import json

import pytest

from vlmfam import tokens as T
from vlmfam.report import RESULTS


def test_fixed_rules_by_hand():
    assert T.llava15(1700, 2200).tokens == 24 * 24
    assert T.gemma3(4032, 3024).tokens == (896 // 14 // 4) ** 2 == 256
    assert T.qwen_vl(10, 10).tokens == 256


def test_smart_resize_rounds_to_28_and_respects_caps():
    for w, h in [(1700, 2200), (4032, 3024), (100, 800), (30, 30)]:
        rh, rw = T.smart_resize(h, w)
        assert rh % 28 == 0 and rw % 28 == 0
        assert 56 * 56 <= rh * rw <= 16384 * 28 * 28
    assert T.qwen2_vl(224, 224).tokens == 64  # 8 x 8 merged tokens (the post's 66 adds the two vision markers)


def test_tile_grids_and_internvl_choice():
    grids = T.tile_grids(1, 12)
    assert len(grids) == sum(1 for c in range(1, 13) for r in range(1, 13) if c * r <= 12)
    assert T.internvl_grid(1700, 2200) == (3, 4)  # aspect 0.77, closest grid 3:4
    assert T.internvl(300, 300).tokens == 256  # one tile, no thumbnail
    assert T.internvl(1700, 2200).tokens == (12 + 1) * 256


def test_llava_next_unpadding_beats_the_naive_count_on_wide_images():
    c = T.llava_next(1920, 1080)
    assert c.tokens == c.parts["base"] + c.parts["tiles"] + c.parts["newlines"]
    assert c.tokens < c.parts["naive"]


def test_pixtral_counts_row_breaks():
    c = T.pixtral(160, 80)
    assert c.grid == (10, 5) and c.tokens == 5 * (10 + 1)


def test_idefics3_upscales_small_images():
    assert T.idefics3(300, 300).tokens == (4 * 4 + 1) * 169  # long side resized up to 1,456 px
    assert T.post_formula("Idefics3", 300, 300) == 169


def test_rules_reproduce_recorded_processor_outputs():
    """results/02_check_processors.json holds what the real HF processors returned."""
    from vlmfam import processors as P

    path = RESULTS / "02_check_processors.json"
    if not path.exists():
        pytest.skip("run scripts/02_check_processors.py first")
    defaults = {"LLaVA-1.5": T.llava15, "LLaVA-NeXT": T.llava_next, "LLaVA-OneVision": T.llava_onevision,
                "InternVL 1.5-3": T.internvl, "Pixtral": T.pixtral, "Gemma 3": T.gemma3, "Qwen2/2.5-VL": T.qwen2_vl}
    smol = {"HuggingFaceTB/SmolVLM-256M-Instruct": dict(longest_edge=2048, tile=512, tokens_per_tile=64)}
    n = 0
    for row in json.loads(path.read_text())["data"]:
        rule = defaults.get(row["rule"])
        if row["rule"] == "Idefics3":
            kw = smol.get(row["repo"], {})
            rule = lambda w, h, kw=kw: T.idefics3(w, h, **kw)  # noqa: E731
        for s in row["sizes"]:
            assert rule(s["width"], s["height"]).tokens == s["processor"], (row["repo"], s)
            n += 1
    assert n == len(P.PROCESSORS) * len(P.SIZES)
