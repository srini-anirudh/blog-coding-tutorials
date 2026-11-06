"""Projector shapes, pixel shuffle against a loop, and gradients reaching only the projector."""

import pytest
import torch

from tinyvlm.projector import LinearProjector, MLPProjector, pixel_shuffle
from tinyvlm.testing import tiny_vlm


@pytest.mark.parametrize("cls", [LinearProjector, MLPProjector])
def test_projector_shapes(cls):
    assert cls(768, 576)(torch.randn(3, 196, 768)).shape == (3, 196, 576)


def test_pixel_shuffle_matches_loop():
    n, side, d, f = 2, 4, 3, 2
    x = torch.randn(n, side * side, d)
    grid = x.reshape(n, side, side, d)
    expected = torch.stack(
        [
            torch.cat([grid[:, f * i + a, f * j + b] for a in range(f) for b in range(f)], dim=-1)
            for i in range(side // f)
            for j in range(side // f)
        ],
        dim=1,
    )
    out = pixel_shuffle(x, f)
    assert out.shape == (n, 4, d * 4)
    # Same 4 patches per token; the channel order inside a token differs, so compare as sets.
    assert torch.allclose(out.sort(-1).values, expected.sort(-1).values)


def test_pixel_shuffle_equals_idefics3():
    """Exact match with the Hugging Face implementation that SmolVLM uses, channel order included."""
    from transformers.models.idefics3.modeling_idefics3 import Idefics3Connector

    x = torch.randn(2, 196, 12)
    assert torch.equal(pixel_shuffle(x, 2), Idefics3Connector.pixel_shuffle(None, x, 2))


def test_pixel_shuffle_rejects_bad_grid():
    with pytest.raises(ValueError):
        pixel_shuffle(torch.randn(1, 15, 8), 2)


def test_only_projector_gets_gradients():
    model, proc = tiny_vlm()
    model.set_trainable(vision=False, projector=True, lm=False)
    from PIL import Image

    msgs = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": "red"},
    ]
    batch = proc.collate([proc.build_example(msgs, [Image.new("RGB", (40, 30), "red")])])
    batch.pop("num_crops"), batch.pop("image_index")
    model(**batch).loss.backward()
    assert all(p.grad is not None for p in model.projector.parameters())
    assert all(p.grad is None for p in model.vision.parameters())
    assert all(p.grad is None for p in model.lm.parameters())
    counts = model.parameter_counts()
    assert counts["trainable"] == counts["projector"]
