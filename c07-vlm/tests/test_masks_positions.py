"""Attention masks and position ids."""

import torch

from tinyvlm.masks import build_attention_mask, to_additive
from tinyvlm.positions import position_ids_1d, position_ids_3d
from tinyvlm.testing import tiny_vlm


def test_causal_and_padding():
    mask = torch.tensor([[0, 1, 1, 1]])  # one left-pad slot
    allowed = build_attention_mask(mask)[0, 0]
    assert allowed.shape == (4, 4)
    assert not allowed[:, 0].any()  # nobody attends to padding
    assert allowed[3, 1:].all() and not allowed[1, 2]  # causal


def test_bidirectional_image_block():
    mask = torch.ones(1, 5, dtype=torch.long)
    index = torch.tensor([[-1, 0, 0, 0, -1]])
    allowed = build_attention_mask(mask, index, bidirectional_images=True)[0, 0]
    assert allowed[1, 3]  # first image token sees the last one
    assert not allowed[0, 1]  # text before the image still cannot see it
    assert not build_attention_mask(mask, index)[0, 0][1, 3]  # default stays causal


def test_explicit_4d_mask_matches_hf_default():
    model, _ = tiny_vlm()
    ids = torch.randint(4, 100, (2, 6))
    attn = torch.tensor([[1] * 6, [0, 0, 1, 1, 1, 1]])
    pos = position_ids_1d(attn)
    ref = model.lm(input_ids=ids, attention_mask=attn, position_ids=pos).logits
    ours = model.lm(input_ids=ids, attention_mask=to_additive(build_attention_mask(attn)), position_ids=pos).logits
    real = attn.bool()
    assert torch.allclose(ref[real], ours[real], atol=1e-5)


def test_position_ids_1d_left_padding():
    pos = position_ids_1d(torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]))
    assert pos.tolist() == [[1, 1, 0, 1, 2], [0, 1, 2, 3, 4]]


def test_position_ids_3d_compresses_image_extent():
    index = torch.tensor([-1, -1] + [0] * 6 + [-1])  # 2 text, a 2 x 3 image, 1 text
    pos = position_ids_3d(index, grid=(2, 3))
    assert pos[:, :2].tolist() == [[0, 1]] * 3
    assert pos[0, 2:8].tolist() == [2] * 6  # one temporal index
    assert pos[1, 2:8].tolist() == [2, 2, 2, 3, 3, 3]  # rows
    assert pos[2, 2:8].tolist() == [2, 3, 4, 2, 3, 4]  # columns
    assert pos[:, 8].tolist() == [5, 5, 5]  # text resumes at 2 + max(2, 3), not at 2 + 6
