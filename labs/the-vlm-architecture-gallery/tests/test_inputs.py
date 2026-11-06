import itertools
from functools import partial
from types import SimpleNamespace

import pytest

from gallery import inputs

SIZES = [(768, 1024), (3024, 4032), (2000, 600), (1754, 1240), (100, 100), (30, 2000), (448, 448), (5000, 5000)]
PINPOINTS = [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]


def test_fixed_square_llava15():
    assert inputs.fixed_square(336, 14)["tokens"] == 576


@pytest.mark.parametrize("hw", SIZES)
def test_smart_resize_matches_hf(hw):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize

    assert inputs.smart_resize(*hw) == smart_resize(*hw)
    assert inputs.smart_resize(*hw, max_pixels=12_845_056) == smart_resize(*hw, max_pixels=12_845_056)


def test_native_token_range_matches_qwen2vl_config():
    assert inputs.native_tokens((10, 10))["tokens"] == 4                  # min_pixels / 28^2
    assert inputs.native_tokens((20000, 20000))["tokens"] <= 16384       # max_pixels / 28^2


@pytest.mark.parametrize("hw", SIZES)
def test_select_best_resolution_matches_hf(hw):
    from transformers.image_processing_utils import select_best_resolution

    assert inputs.select_best_resolution(hw, PINPOINTS) == tuple(select_best_resolution(hw, PINPOINTS))


@pytest.mark.parametrize("hw", SIZES)
def test_llava_next_feature_count_matches_hf(hw):
    from transformers.models.llava_next.processing_llava_next import LlavaNextProcessor

    fake = SimpleNamespace(image_processor=SimpleNamespace(image_grid_pinpoints=[list(p) for p in PINPOINTS]),
                           patch_size=14, num_additional_image_tokens=0)
    fake._get_unpadded_features = partial(LlavaNextProcessor._get_unpadded_features, fake)
    hf = LlavaNextProcessor._get_number_of_features(fake, hw[0], hw[1], 336, 336)
    assert inputs.llava_next_tokens(hw, PINPOINTS)["tokens"] == hf


@pytest.mark.parametrize("hw,max_tiles", list(itertools.product(SIZES, [6, 12, 40])))
def test_internvl_grid_matches_hf_got_ocr2(hw, max_tiles):
    from transformers.models.got_ocr2.image_processing_pil_got_ocr2 import get_optimal_tiled_canvas

    hf = get_optimal_tiled_canvas(hw, (448, 448), 1, max_tiles)
    assert inputs.closest_grid(hw, 448, 1, max_tiles) == tuple(hf)


def test_internvl_forty_tiles_is_over_ten_thousand():
    r = inputs.internvl_tokens((4000, 6400), max_tiles=40)
    assert r["grid"] == (8, 5) and r["tiles"] == 41 and r["tokens"] == 10496


def test_multicrop_overlap_costs_tokens():
    no = inputs.multicrop_tokens((1008, 1008), overlap=0, global_view=False)
    yes = inputs.multicrop_tokens((1008, 1008), overlap=56, global_view=False)
    assert no["grid"] == (3, 3) and no["pixels_encoded_over_covered"] == 1.0
    assert yes["tokens"] > no["tokens"] and yes["pixels_encoded_over_covered"] > 1.0
