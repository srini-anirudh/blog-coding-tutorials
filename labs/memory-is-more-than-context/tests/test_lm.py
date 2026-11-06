"""Real-model tests. They need Qwen2.5-0.5B-Instruct from the Hugging Face Hub."""

from __future__ import annotations

import random

import pytest
import torch

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def lm():
    from memlab.lm import LM

    return LM(eager=True)


def test_greedy_check_agrees_with_generate(lm):
    from memlab.position import kv_case

    c = kv_case(5, 2, random.Random(0))
    ids = torch.tensor([lm.prompt_ids(c.prompt)])
    with torch.no_grad():
        gen = lm.model.generate(ids, max_new_tokens=48, do_sample=False)
    text = lm.tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    check = lm.greedy_matches(c.prompt, [c.answer, f'"{c.answer}"'], needle=c.needle)
    assert check.correct == (text.strip().strip('"').startswith(c.answer))
    assert 0.0 < check.needle_attention < 1.0


def test_wrong_answer_is_not_matched(lm):
    check = lm.greedy_matches("What is 2 + 2? Reply with the number only.", "5")
    assert not check.correct and check.logprob < 0
