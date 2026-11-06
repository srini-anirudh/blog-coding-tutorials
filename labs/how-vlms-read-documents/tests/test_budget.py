import pytest

from docread.budget import (deepseek_mode_tokens, effective_dpi_fixed, native_tokens, post_formula_tokens,
                            px_per_em, smart_resize, tiling_tokens_figure, tiling_tokens_internvl)


@pytest.mark.parametrize("dpi, post", [(72, 640), (150, 2780), (200, 4930), (300, 11100)])
def test_post_formula(dpi, post):
    assert post_formula_tokens(dpi) == pytest.approx(post, rel=0.005)
    assert native_tokens(dpi) == pytest.approx(post, rel=0.02)


def test_smart_resize_rounds_to_28_and_caps_area():
    h, w = smart_resize(842, 595)
    assert h % 28 == 0 and w % 28 == 0
    h, w = smart_resize(20000, 14000)
    assert h * w <= 12845056


def test_fixed_modes_and_glyph_size():
    assert [deepseek_mode_tokens(s) for s in (512, 640, 1024, 1280)] == [64, 100, 256, 400]
    assert px_per_em(10, 72) == 10 and px_per_em(10, 150) == pytest.approx(20.8, abs=0.1)
    assert effective_dpi_fixed(1024) == pytest.approx(87.6, abs=0.1)


def test_tiling():
    assert tiling_tokens_figure(300) == (40, 41 * 256)  # the cap
    assert tiling_tokens_internvl(72)[0] == tiling_tokens_internvl(300)[0] == 35  # 5 x 7 grid, any dpi
    assert tiling_tokens_internvl(200, max_tiles=12)[0] == 6


@pytest.mark.network
def test_real_qwen2_vl_processor_agrees():
    from PIL import Image
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    for dpi in (72, 200):
        img = Image.new("RGB", (round(8.27 * dpi), round(11.69 * dpi)), "white")
        t, gh, gw = proc(images=img, return_tensors="pt")["image_grid_thw"][0].tolist()
        assert t * gh * gw // 4 == native_tokens(dpi)
