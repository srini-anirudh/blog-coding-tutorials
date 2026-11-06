"""Checks against real checkpoints and processors. They download from the Hugging Face Hub the first time."""

import numpy as np
import pytest
import torch
from PIL import Image

from seelab.tokens import smart_resize

pytestmark = pytest.mark.network


@pytest.mark.parametrize("size", [(1000, 1300), (1920, 1080), (300, 2000)])
def test_smart_resize_matches_qwen2vl_processor(size):
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    w, h = size
    t, gh, gw = proc(images=Image.fromarray(np.zeros((h, w, 3), np.uint8)), return_tensors="pt")["image_grid_thw"][0].tolist()
    rh, rw = smart_resize(h, w, min_pixels=proc.size["shortest_edge"], max_pixels=proc.size["longest_edge"])
    assert (gh, gw) == (rh // 14, rw // 14)


@pytest.mark.slow
def test_zero_gates_on_real_smollm2():
    from seelab.gated import FlamingoLite
    from seelab.vlm import build

    vlm, processor = build("mlp")
    ids = processor.tokenizer(["Three orange squares."], return_tensors="pt").input_ids
    with torch.no_grad():
        before = vlm.lm(input_ids=ids).logits
        fl = FlamingoLite(vlm.vision, vlm.lm)
        after = fl(ids, pixel_values=torch.randn(1, 3, 224, 224)).logits
    assert torch.equal(before, after)
