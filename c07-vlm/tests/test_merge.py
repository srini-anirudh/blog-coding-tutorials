"""Scattering image features into the embedding sequence."""

import pytest
import torch

from tinyvlm.merge import merge_masked_scatter, merge_naive

IMG = 7


def make_inputs():
    ids = torch.tensor([[1, IMG, IMG, 2, IMG, IMG, 3], [4, 5, IMG, IMG, 6, 0, 0]])
    embeds = torch.randn(2, 7, 5)
    features = torch.randn(3, 2, 5)  # 3 crops of 2 tokens; crops 0, 1 belong to sample 0
    return ids, embeds, features


def test_naive_equals_masked_scatter():
    ids, embeds, features = make_inputs()
    assert torch.equal(merge_naive(embeds, ids, features, IMG), merge_masked_scatter(embeds, ids, features, IMG))


def test_features_land_in_order_and_text_is_untouched():
    ids, embeds, features = make_inputs()
    out = merge_masked_scatter(embeds, ids, features, IMG)
    assert torch.equal(out[0, 1:3], features[0])
    assert torch.equal(out[0, 4:6], features[1])
    assert torch.equal(out[1, 2:4], features[2])
    text = ids != IMG
    assert torch.equal(out[text], embeds[text])


def test_count_mismatch_raises():
    ids, embeds, features = make_inputs()
    with pytest.raises(ValueError, match="placeholders"):
        merge_masked_scatter(embeds, ids, features[:2], IMG)


def test_gradient_flows_to_features():
    ids, embeds, features = make_inputs()
    features.requires_grad_(True)
    merge_masked_scatter(embeds, ids, features, IMG).sum().backward()
    assert torch.equal(features.grad, torch.ones_like(features))
