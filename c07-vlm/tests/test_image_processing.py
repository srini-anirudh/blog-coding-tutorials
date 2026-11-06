"""The image processor matches Hugging Face's SigLIP processor and the resolution strategies count crops right."""

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.models.siglip.image_processing_pil_siglip import SiglipImageProcessorPil

from tinyvlm.image_processing import TinyVLMImageProcessor, normalize, pad_to_square, select_grid


def random_image(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize("size", [(224, 224), (640, 480), (97, 301)])
def test_matches_hf_siglip_processor(size):
    image = random_image(*size)
    ours = TinyVLMImageProcessor()([image])["pixel_values"]
    hf = SiglipImageProcessorPil(size={"height": 224, "width": 224}, resample=3)(image, return_tensors="pt")["pixel_values"]
    assert ours.shape == hf.shape == (1, 3, 224, 224)
    assert (ours - hf).abs().max() < 1e-5


def test_normalize_range_and_layout():
    black, white = Image.new("RGB", (4, 2), (0, 0, 0)), Image.new("RGB", (4, 2), (255, 255, 255))
    assert normalize(black).shape == (3, 2, 4)  # channels first, then H, then W
    assert torch.allclose(normalize(black), torch.full((3, 2, 4), -1.0))
    assert torch.allclose(normalize(white), torch.full((3, 2, 4), 1.0))


def test_padding_is_zero_after_normalisation():
    padded = pad_to_square(Image.new("RGB", (200, 100), (255, 0, 0)))
    assert padded.size == (200, 200)
    corner = normalize(padded)[:, 0, 0]
    assert corner.abs().max() < 0.01  # mean colour 127/128 -> about 0


@pytest.mark.parametrize(
    "size, grid",
    [((224, 224), (1, 1)), ((448, 224), (2, 1)), ((1000, 300), (3, 1)), ((300, 1000), (1, 3)), ((1344, 896), (3, 2)), ((100, 50), (2, 1))],
)
def test_select_grid(size, grid):
    assert select_grid(*size, tile_size=224, max_tiles=6) == grid


@pytest.mark.parametrize("size", [(224, 224), (640, 480), (1000, 300), (1344, 896)])
def test_tile_count_matches_prediction(size):
    proc = TinyVLMImageProcessor(strategy="tile", max_tiles=6)
    out = proc([random_image(*size)])
    assert out["num_crops"] == [proc.num_crops_for_size(*size)]
    assert out["pixel_values"].shape == (out["num_crops"][0], 3, 224, 224)


def test_save_and_load(tmp_path):
    proc = TinyVLMImageProcessor(strategy="tile", max_tiles=4)
    proc.save_pretrained(tmp_path)
    again = TinyVLMImageProcessor.from_pretrained(tmp_path)
    assert (again.strategy, again.max_tiles, again.size) == ("tile", 4, 224)
