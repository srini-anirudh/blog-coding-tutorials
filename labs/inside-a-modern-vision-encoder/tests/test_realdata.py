"""Real-model helpers. The offline tests use synthetic tensors; the network tests load DINOv2 and a Hub config."""

import json

import pytest
import torch
from PIL import Image

from vencoder.realdata import dinov2_input, knn_accuracy, outlier_stats


def test_outlier_stats():
    norms = torch.ones(2, 100)
    norms[0, 3] = 10.0
    s = outlier_stats(norms, 3.0)
    assert s["outlier_fraction"] == pytest.approx(1 / 200)
    assert s["images_with_outliers"] == 0.5 and s["max_over_median_max"] == pytest.approx(10.0)


def test_knn_accuracy_separable():
    feats = torch.cat([torch.randn(20, 4) * 0.1 + 3, torch.randn(20, 4) * 0.1 - 3])
    labels = torch.tensor([0] * 20 + [1] * 20)
    assert knn_accuracy(feats, labels, k=5) == 1.0


def test_dinov2_input_shapes_and_norm():
    img = Image.new("RGB", (640, 480), (124, 116, 104))  # roughly the ImageNet mean
    x = dinov2_input(img)
    assert x.shape == (3, 224, 224) and x.abs().max() < 0.05
    assert dinov2_input(img, crop=False).shape == (3, 224, 224)
    assert dinov2_input(img, resize="torch_no_aa").shape == (3, 224, 224)


@pytest.mark.network
def test_dinov2_registers_config():
    from vencoder.realdata import load_dinov2

    model, n_reg = load_dinov2("facebook/dinov2-with-registers-small")
    assert n_reg == 4
    with torch.no_grad():
        o = model(pixel_values=torch.zeros(1, 3, 224, 224))
    assert o.last_hidden_state.shape[1] == 1 + 4 + 256


@pytest.mark.network
def test_qwen25_vit_constants_match_hub():
    from huggingface_hub import hf_hub_download

    c = json.load(open(hf_hub_download("Qwen/Qwen2.5-VL-3B-Instruct", "config.json")))["vision_config"]
    p = json.load(open(hf_hub_download("Qwen/Qwen2.5-VL-3B-Instruct", "preprocessor_config.json")))
    assert (c["hidden_size"], c["intermediate_size"], c["depth"], c["window_size"]) == (1280, 3420, 32, 112)
    assert list(c["fullatt_block_indexes"]) == [7, 15, 23, 31]
    from vencoder.intake import QWEN_MAX_PIXELS, QWEN_MIN_PIXELS

    assert (p["min_pixels"], p["max_pixels"]) == (QWEN_MIN_PIXELS, QWEN_MAX_PIXELS)
