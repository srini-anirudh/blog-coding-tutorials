"""Tests on the real SmolVLM-256M-Instruct. They download about 1 GB the first time, so they are marked `network`."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmscope import scenes  # noqa: E402
from vlmscope.intervene import Intervention  # noqa: E402

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def vlm():
    from vlmscope.model import VLM
    return VLM(device="cpu")


@pytest.fixture(scope="module")
def prompt(vlm):
    shapes = [scenes.Shape("circle", "red", 64, 64, 160), scenes.Shape("square", "blue", 300, 300, 150)]
    return vlm.prompt(scenes.render(shapes), "What colour is the circle?")


def test_prompt_layout(vlm, prompt):
    assert len(prompt.image_pos) == 64
    assert prompt.question_pos[0] > prompt.image_pos[-1]
    assert prompt.tokens[prompt.find("circle")].strip("Ġ") == "circle"
    assert prompt.last == prompt.input_ids.shape[1] - 1


def test_lab_attention_matches_sdpa(vlm, prompt):
    from transformers import AutoModelForImageTextToText
    ref = AutoModelForImageTextToText.from_pretrained(vlm.model.name_or_path, dtype=torch.float32,
                                                      attn_implementation="sdpa").eval()
    with torch.no_grad():
        want = ref(input_ids=prompt.input_ids, **prompt.extra).logits[0, -1]
    assert (vlm.run(prompt) - want).abs().max() < 1e-3


def test_answers_red_and_knockout_removes_it(vlm, prompt):
    ids = vlm.word_ids(list(scenes.COLOURS))
    assert int(vlm.run(prompt)[ids].argmax()) == 0
    everyone = torch.arange(prompt.input_ids.shape[1])
    ko = {l: [(everyone, prompt.image_pos)] for l in range(vlm.n_layers)}
    p = torch.softmax(vlm.run(prompt, Intervention(knockout=ko)), -1)[ids[0]]
    assert p < 0.5


def test_connector_tokens_are_raster_blocks(vlm):
    idx = torch.arange(1024.0).view(1, 1024, 1)
    shuffled = vlm.connector.pixel_shuffle(idx, vlm.connector.scale_factor)[0]
    # token 21 = grid row 2, col 5 = patch rows 8..11, patch cols 20..23 of the 32 x 32 patch grid
    want = sorted(r * 32 + c for r in range(8, 12) for c in range(20, 24))
    assert sorted(shuffled[21].long().tolist()) == want


def test_embeddings_hold_connector_output(vlm, prompt):
    shapes = [scenes.Shape("circle", "red", 64, 64, 160), scenes.Shape("square", "blue", 300, 300, 150)]
    _, projected = vlm.vision_features(scenes.render(shapes))
    assert torch.allclose(vlm.embeddings(prompt)[prompt.image_pos], projected, atol=1e-5)
