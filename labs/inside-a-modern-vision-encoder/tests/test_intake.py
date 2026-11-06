"""Intake strategies: our re-implementations against the reference functions, and the post's token counts."""

import ast
import inspect
import math

import pytest

from vencoder import intake
from vencoder.intake import (
    LLAVA_NEXT_PINPOINTS, bench, closest_grid, naflex_size, plan_clip_crop, plan_internvl, plan_qwen, smart_resize,
    select_best_resolution, vit_flops,
)

SIZES = [(1200, 1600), (1024, 768), (320, 1024), (3840, 2160), (600, 3000), (57, 999), (2000, 50), (448, 448)]


def _extract(module, name):
    """Run one top-level function from an installed transformers file without importing the module
    (some of these modules need torchvision, which this repo does not install)."""
    src = inspect.getsource(module) if not isinstance(module, str) else open(module).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    ns = {"math": math}
    exec(compile(ast.Module([fn], []), name, "exec"), ns)
    return ns[name]


def _tf_file(rel):
    import importlib.util
    import os

    spec = importlib.util.find_spec("transformers")
    return os.path.join(os.path.dirname(spec.origin), rel)


@pytest.mark.parametrize("w,h", SIZES)
def test_smart_resize_matches_transformers(w, h):
    ref = _extract(_tf_file("models/qwen2_vl/image_processing_qwen2_vl.py"), "smart_resize")
    for mx in (intake.QWEN_MAX_PIXELS, intake.QWEN_CODE_DEFAULT_MAX_PIXELS):
        assert smart_resize(h, w, 28, 3136, mx) == ref(h, w, 28, 3136, mx)


@pytest.mark.parametrize("w,h", SIZES)
def test_naflex_matches_transformers(w, h):
    ref = _extract(_tf_file("models/siglip2/image_processing_siglip2.py"), "get_image_size_for_max_num_patches")
    assert naflex_size(h, w, 16, 1024) == ref(h, w, 16, 1024)


@pytest.mark.parametrize("w,h", SIZES)
def test_internvl_grid_matches_got_ocr2(w, h):
    from transformers.models.got_ocr2.image_processing_pil_got_ocr2 import get_optimal_tiled_canvas

    assert closest_grid(w, h, 448, 1, 12) == tuple(get_optimal_tiled_canvas((h, w), (448, 448), 1, 12))


@pytest.mark.parametrize("w,h", SIZES)
def test_anyres_matches_transformers(w, h):
    from transformers.image_processing_utils import select_best_resolution as ref

    assert select_best_resolution((h, w), LLAVA_NEXT_PINPOINTS) == tuple(ref((h, w), LLAVA_NEXT_PINPOINTS))


def test_post_token_counts():
    assert intake.plan_squash(999, 333).patches == 576  # CLIP ViT-L/14 @336
    assert plan_internvl(448, 448).patches == 1024 and plan_internvl(448, 448).llm_tokens == 256
    assert plan_clip_crop(1200, 1600, 224, 16).patches == 196  # DINO v1 on the walkthrough page
    q = plan_qwen(1200, 1600)
    assert q.patches == 9804 and q.llm_tokens == 2451  # the post's Qwen count; 50x the DINO v1 count
    assert plan_qwen(3840, 2160).patches > 20000  # "a 4K frame can exceed 20,000 tokens"


def test_qwen_dims_are_multiples_of_28():
    for w, h in SIZES:
        p = plan_qwen(w, h)
        assert p.resize_to[0] % 28 == 0 and p.resize_to[1] % 28 == 0


def test_internvl_tiles_at_most_12_plus_thumbnail():
    for w, h in SIZES:
        p = plan_internvl(w, h)
        assert 1 <= p.views <= 13


def test_bench_crop_discards_and_squash_distorts():
    r = bench("Receipt 1:3.2", "clip_crop")
    assert r["discarded_pct"] > 60 and r["resolved_measured"] < 5
    s = bench("Receipt 1:3.2", "squash")
    assert s["distortion_pct"] > 200 and s["discarded_pct"] == 0
    n = bench("Receipt 1:3.2", "qwen")
    assert n["resolved_measured"] == 5 and n["distortion_pct"] < 10


def test_window_attention_cheaper():
    full = vit_flops(9804, 1280, 3420, 32)
    win = vit_flops(9804, 1280, 3420, 32, window=64, full_layers=4)
    assert win["projections"] == full["projections"] and win["attention"] < full["attention"] / 3
