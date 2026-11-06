"""Tests that load Qwen2-VL-2B-Instruct (about 4.4 GB from the Hugging Face Hub). Run with: pytest -m network."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = [pytest.mark.network, pytest.mark.slow]


@pytest.fixture(scope="module")
def vlm():
    from lookagain.vlm import QwenVLM
    return QwenVLM(device="cpu", attn="eager")


def test_visual_token_count_matches_the_model(vlm):
    """The tokens we put in the prompt must equal the number of image features the vision tower produces."""
    import torch
    from lookagain.scene import make_scene
    from lookagain.views import cell_view, render
    from lookagain.vlm import n_tokens, patchify
    img = render(make_scene(2000), cell_view(3, 0, width=224))
    rows, grid = patchify(img)
    with torch.no_grad():
        feats = vlm.model.model.visual(rows.to(vlm.model.dtype), grid_thw=grid[None])
    feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
    assert feats.shape[0] == n_tokens(img)


def test_crop_is_read_and_blind_is_not(vlm):
    from lookagain.answers import QUESTION, closing_time
    from lookagain.scene import make_scene
    from lookagain.views import cell_view, render
    s = make_scene(2000)
    crop = closing_time(vlm.ask([render(s, cell_view(3, 0))], QUESTION))
    assert crop == s.close_time
    assert closing_time(vlm.ask([], QUESTION)) != s.close_time


def test_attention_map_shape_and_mass(vlm):
    from lookagain.answers import QUESTION
    from lookagain.scene import make_scene
    from lookagain.views import full_view, render
    a = vlm.attention_map(render(make_scene(2000), full_view(448)), QUESTION)
    assert a.shape == (12, 16) and (a >= 0).all() and a.sum() <= 1.0 + 1e-3
