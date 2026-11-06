"""Checks against the real SmolVLM processor and weights. Downloads from the Hub on first run."""

import pytest
import torch

from vlmalign.dpo import pair_batch
from vlmalign.freeze import MODES, configure_trainable
from vlmalign.logprobs import IGNORE, sequence_logprobs
from vlmalign.model import chat_prompt, encode_answer, load_model, load_processor
from vlmalign.preference import build_pairs

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def proc():
    return load_processor()


def test_one_tile_gives_64_image_tokens(proc):
    b = pair_batch(proc, build_pairs(1, "balanced", seed=1))
    image_id = proc.tokenizer.convert_tokens_to_ids("<image>")
    assert (b.input_ids[0] == image_id).sum() == 64
    assert b.pixel_values.shape == (1, 1, 3, 512, 512)


def test_answer_ids_match_the_chat_template(proc):
    """Our prompt + answer concatenation spells what the template spells for an assistant turn."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "The circle is red."}]}]
    full = proc.apply_chat_template(msgs).strip()
    ours = chat_prompt(proc, "Q?") + proc.tokenizer.decode(encode_answer(proc, "The circle is red."))
    assert ours == full


def test_chosen_and_rejected_share_the_prompt_and_differ_in_the_answer(proc):
    p = build_pairs(1, "balanced", seed=1)[0]
    b = pair_batch(proc, [p])
    n = (b.labels[0] == IGNORE).sum().item()
    assert b.image_index.tolist() == [0, 0]
    prompt_len = (b.labels[0] != IGNORE).nonzero()[0].item()
    assert torch.equal(b.input_ids[0, :prompt_len], b.input_ids[1, :prompt_len])
    assert n > 64


@pytest.mark.slow
def test_lora_regexes_cover_every_block_of_smolvlm():
    model = load_model()
    b = pair_batch(load_processor(), build_pairs(1, "balanced", seed=1))
    before, _ = sequence_logprobs(model, b)
    counts = configure_trainable(model, MODES[-1], r=16)
    # 30 decoder blocks x 7 linears, 12 vision blocks x 6 linears, r = 16
    assert counts["llm"] > 0 and counts["vision"] > 0
    after, _ = sequence_logprobs(model, b)
    torch.testing.assert_close(before, after)
