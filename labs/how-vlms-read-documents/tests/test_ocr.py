import random

import pytest
import torch

from docread.ocr import (CHARSET, CRNN, HEIGHT, WIDTH, downscale, encode, greedy_decode, make_batch, read_lines,
                         render_line, to_tensor, train_crnn)


def test_render_and_tensor_shapes():
    small, big = render_line("1,580.00", 6), render_line("1,580.00", 24)
    assert small.height < big.height
    assert downscale(big, 0.25).height == round(big.height * 0.25)
    x = to_tensor(small)
    assert x.shape == (1, HEIGHT, WIDTH) and 0 <= x.min() and x.max() <= 1


def test_greedy_decode_merges_repeats_and_drops_blanks():
    labels = [0, 2, 2, 0, 2, 11, 0]  # blank 1 1 blank 1 a blank -> "11a"
    logits = torch.full((len(labels), 1, len(CHARSET) + 1), -5.0)
    for t, k in enumerate(labels):
        logits[t, 0, k] = 5.0
    assert greedy_decode(logits) == ["11a"]
    assert encode("10") == [2, 1]


def test_crnn_shapes_and_overfits_one_batch():
    torch.manual_seed(0)
    model = CRNN()
    x, targets, lengths, _ = make_batch(random.Random(0), 8, (14, 20))
    assert model(x).shape == (WIDTH // 4, 8, len(CHARSET) + 1)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    ctc = torch.nn.CTCLoss(zero_infinity=True)
    losses = []
    for _ in range(40):
        out = model(x).log_softmax(-1)
        loss = ctc(out, targets, torch.full((8,), out.shape[0]), lengths)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


@torch.no_grad()
def _acc(model, em, n=40):
    rng = random.Random(7)
    texts = ["".join(rng.choice("0123456789") for _ in range(5)) for _ in range(n)]
    return sum(p == t for p, t in zip(read_lines(model, [render_line(t, em) for t in texts]), texts)) / n


@pytest.mark.slow
def test_short_training_reads_large_text_better_than_tiny_text():
    model = train_crnn(steps=600)
    assert _acc(model, 20) > 0.5
    assert _acc(model, 3) < _acc(model, 20)
