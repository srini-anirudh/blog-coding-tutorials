"""Masks: every pattern checked against its definition."""

import torch

from attnzoo import masks as M


def test_causal_is_lower_triangular():
    assert torch.equal(M.causal_mask(5), torch.ones(5, 5, dtype=torch.bool).tril())


def test_causal_bottom_right_alignment_for_decoding():
    # One new query after 4 cached tokens sees all 5 keys, not just key 0.
    assert M.causal_mask(1, 5).all()
    m = M.causal_mask(2, 5)
    assert m[0].tolist() == [True, True, True, True, False]


def test_sliding_window_sees_exactly_w_keys():
    m = M.sliding_window_mask(10, window=3)
    assert m.sum(-1).tolist() == [1, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    assert not m[9, 6] and m[9, 7]


def test_padding_masks_keys_only():
    pm = M.padding_mask(torch.tensor([[1, 1, 1, 0]]))
    assert pm.shape == (1, 1, 1, 4) and pm[0, 0, 0].tolist() == [True, True, True, False]


def test_combine_is_and():
    c = M.combine(M.causal_mask(4)[None, None], M.padding_mask(torch.tensor([[0, 1, 1, 1]])))
    assert c.shape == (1, 1, 4, 4)
    assert not c[0, 0, 3, 0] and c[0, 0, 3, 1]


def test_prefix_lm():
    m = M.prefix_lm_mask(5, 2)
    assert m[0, 1] and not m[2, 3] and m[4].all()


def test_document_mask_blocks_across_documents():
    m = M.document_mask(torch.tensor([[0, 0, 1, 1, 1]]))[0, 0]
    assert m[1, 0] and not m[2, 1] and m[4, 2] and not m[2, 4]


def test_multimodal_mask_image_bidirectional_text_causal():
    span = torch.tensor([[0, 1, 1, 1, 0, 0]])  # text, 3 image tokens, text, text
    m = M.multimodal_mask(span)[0, 0]
    assert m[1, 3]  # image patch sees a later patch of the same image
    assert not m[0, 1]  # text before the image cannot see it
    assert m[4, 1:4].all() and not m[4, 5]  # text after sees the image, but not the future


def test_local_global():
    m = M.local_global_mask(8, window=2, global_positions=[0])
    assert m[7, 0] and m[0, 7] and m[5, 4] and not m[5, 2]


def test_block_sparse_expands():
    layout = M.block_sparse_layout(4, local_blocks=1, global_blocks=1)
    tok = M.expand_block_layout(layout, 2)
    assert tok.shape == (8, 8) and tok[6, 0] and not tok[6, 2]


def test_mask_to_bias():
    b = M.mask_to_bias(M.causal_mask(3))
    assert b[0, 0] == 0 and torch.isinf(b[0, 1])


def test_structural_mask_none_when_bidirectional():
    assert M.structural_mask(4, 4, causal=False, window_size=None) is None
