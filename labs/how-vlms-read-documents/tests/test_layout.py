import random

import pytest
import torch

from docread.layout import (ORDERS, VOCAB, WORD_ID, LayoutEncoder, accuracy, collate, encode_receipt, make_split,
                            ocr_order, synthetic_receipt, train_layout)


def test_total_index_points_at_the_amount_on_the_total_line():
    rng = random.Random(0)
    for _ in range(50):
        words, total = synthetic_receipt(rng)
        assert words[total][0] == "[NUM]"
        key = [w for w, b in words if w == "TOTAL"]
        assert len(key) == 1
        key_box = next(b for w, b in words if w == "TOTAL")
        assert key_box[1] == words[total][1][1]  # same line


def test_orders_are_permutations_and_label_follows():
    rng = random.Random(1)
    words, total = synthetic_receipt(rng)
    for how in ORDERS:
        order = ocr_order(words, how, rng)
        assert sorted(order) == list(range(len(words)))
        ids, boxes, label = encode_receipt(words, total, order)
        assert ids[label] == WORD_ID["[NUM]"] and boxes[label] == words[total][1]
    ids, _, _ = encode_receipt(words, total, list(range(len(words))), ocr_typo=True)
    assert WORD_ID["TOTAL"] not in ids and WORD_ID["[UNK]"] in ids


def test_encoder_shapes_and_padding_mask():
    ids, boxes, mask, label = collate(make_split(4, seed=0))
    for use_2d in (False, True):
        scores = LayoutEncoder(use_2d)(ids, boxes, mask)
        assert scores.shape == ids.shape
        assert (scores[~mask] < -1e8).all()
    assert len(VOCAB) == len(set(VOCAB))


@pytest.mark.slow
def test_boxes_beat_1d_on_shuffled_words():
    torch.manual_seed(0)
    train = make_split(3000, seed=0)
    test = make_split(300, seed=1, orders=["shuffled"])
    acc_1d = accuracy(train_layout(False, train, steps=1500, lr=2e-3), test)
    acc_2d = accuracy(train_layout(True, train, steps=1500, lr=2e-3), test)
    assert acc_2d > acc_1d + 0.3
