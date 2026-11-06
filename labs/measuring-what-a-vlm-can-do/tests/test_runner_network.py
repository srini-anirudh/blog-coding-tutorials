"""Real data and a real model. These download from the Hugging Face Hub."""

import pytest
import torch

from benchlab import runner
from benchlab.mc import extract, format_prompt


@pytest.mark.network
def test_mmstar_loads_and_parses():
    rows, skipped = runner.load_mmstar()
    assert len(rows) + skipped == 1500
    assert skipped == 0
    assert sum(len(r.item.options) == 4 for r in rows) == 1499  # one item has three options


@pytest.mark.network
def test_pope_balanced():
    rows = runner.load_pope("adversarial")
    assert len(rows) == 3000
    assert sum(r["answer"] == "yes" for r in rows) == 1500


@pytest.mark.network
@pytest.mark.slow
def test_smolvlm_answers_two_items_on_cpu():
    rows, _ = runner.load_mmstar(limit=2)
    vlm = runner.VLM(runner.MODELS["256m"][0], torch.device("cpu"), split=False)
    replies, tok = vlm.generate([format_prompt(r.item) for r in rows], [r.image for r in rows], max_new_tokens=8)
    assert tok == 64  # one 512 px view, 64 image tokens after pixel shuffle
    assert all(extract(x, r.item, "content") is not None for x, r in zip(replies, rows))


@pytest.mark.network
@pytest.mark.slow
def test_worker_processes_give_the_same_replies():
    rows = runner.load_pope("random", limit=4)
    vlm = runner.VLM(runner.MODELS["256m"][0], torch.device("cpu"), split=False)
    texts, imgs = [r["question"] for r in rows], [r["image"] for r in rows]
    a, _ = vlm.generate(texts, imgs, max_new_tokens=3, batch_size=2, workers=0)
    b, _ = vlm.generate(texts, imgs, max_new_tokens=3, batch_size=2, workers=2)
    assert a == b
