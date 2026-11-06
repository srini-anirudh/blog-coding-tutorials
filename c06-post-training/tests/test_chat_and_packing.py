"""Stage 1 checks: the chat template, the assistant-only mask, and packing without cross-talk."""

import random

import pytest
import torch

from posttrain.chat import IGNORE_INDEX, ChatExample, encode_chat, load_tokenizer
from posttrain.models import tiny_llama
from posttrain.packing import block_causal_mask, collate_packed, pack_examples, pad_batch
from posttrain.sft import masked_lm_loss
from posttrain.tasks import add_follow_up, make_problem, to_messages


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


@pytest.fixture(scope="module")
def two_turn():
    p = make_problem(random.Random(0), "chain")
    return to_messages(add_follow_up(random.Random(1), p))


@pytest.mark.network
def test_manual_ids_match_chat_template(tok, two_turn):
    ex = encode_chat(tok, two_turn)
    text = tok.apply_chat_template(two_turn, tokenize=False)
    assert ex.input_ids == tok(text, add_special_tokens=False)["input_ids"]


@pytest.mark.network
def test_assistant_mask_matches_hf_generation_mask(tok, two_turn):
    ex = encode_chat(tok, two_turn)
    hf = tok.apply_chat_template(two_turn, return_assistant_tokens_mask=True, return_dict=True)
    assert hf["assistant_masks"] == [int(lab != IGNORE_INDEX) for lab in ex.labels]


@pytest.mark.network
def test_only_assistant_content_and_im_end_are_targets(tok, two_turn):
    ex = encode_chat(tok, two_turn)
    targets = tok.decode([t for t, lab in zip(ex.input_ids, ex.labels) if lab != IGNORE_INDEX])
    replies = [m["content"] + "<|im_end|>" for m in two_turn if m["role"] == "assistant"]
    assert targets == "".join(replies)
    assert all(r in ("assistant",) for r, lab in zip(ex.roles, ex.labels) if lab != IGNORE_INDEX)


@pytest.mark.network
def test_last_turn_mode_trains_one_reply(tok, two_turn):
    ex = encode_chat(tok, two_turn, mode="last")
    targets = tok.decode([t for t, lab in zip(ex.input_ids, ex.labels) if lab != IGNORE_INDEX])
    assert targets == two_turn[-1]["content"] + "<|im_end|>"


@pytest.mark.network
def test_pad_token_is_not_end_of_turn(tok):
    # If pad == <|im_end|>, a collator that masks pad ids would also mask the stop token.
    assert tok.pad_token != "<|im_end|>"


def _ex(ids):
    return ChatExample(list(ids), list(ids), ["assistant"] * len(ids))


def test_packing_keeps_conversations_whole_and_resets_positions():
    exs = [_ex(range(3, 3 + n)) for n in (10, 7, 12, 5, 9)]
    rows = pack_examples(exs, max_len=24)
    assert sorted(len(e.input_ids) for r in rows for e in [r]) and sum(len(r.input_ids) for r in rows) == 43
    for r in rows:
        assert len(r.input_ids) <= 24
        starts = [i for i, p in enumerate(r.position_ids) if p == 0]
        assert all(r.labels[s] == IGNORE_INDEX for s in starts)  # no prediction across a boundary


def test_block_mask_is_causal_and_block_diagonal():
    seq = torch.tensor([[0, 0, 1, 1, 1, -1]])
    m = block_causal_mask(seq)[0, 0]
    assert m.shape == (6, 6)
    assert not m[3, 1]  # token of conv 1 cannot see conv 0
    assert m[4, 2] and not m[2, 4]  # causal inside conv 1
    assert m[5, 5] and m[5].sum() == 1  # padding sees only itself, so softmax stays finite


def test_packed_logits_equal_unpacked_logits():
    model = tiny_llama()
    a, b = _ex(range(3, 13)), _ex(range(20, 27))
    batch = collate_packed(pack_examples([a, b], 24), 24, pad_id=0)
    with torch.no_grad():
        packed = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       position_ids=batch["position_ids"], use_cache=False).logits
        alone_b = model(input_ids=torch.tensor([b.input_ids])).logits
    torch.testing.assert_close(packed[0, 10:17], alone_b[0], atol=1e-5, rtol=1e-5)


def test_naive_packing_leaks_across_conversations():
    model = tiny_llama()
    a, b = _ex(range(3, 13)), _ex(range(20, 27))
    ids = torch.tensor([a.input_ids + b.input_ids])
    with torch.no_grad():
        naive = model(input_ids=ids).logits  # one long causal sequence: b attends to a
        alone_b = model(input_ids=torch.tensor([b.input_ids])).logits
    assert (naive[0, 10:] - alone_b[0]).abs().max() > 1e-2


def test_position_ids_alone_need_use_cache_false():
    """HF builds the block mask from position-id resets, but only when no KV cache is created."""
    model = tiny_llama()
    a, b = _ex(range(3, 13)), _ex(range(20, 27))
    batch = collate_packed(pack_examples([a, b], 24), 24, pad_id=0)
    with torch.no_grad():
        alone_b = model(input_ids=torch.tensor([b.input_ids])).logits[0]
        no_cache = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"], use_cache=False).logits
        cache = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"], use_cache=True).logits
    assert (no_cache[0, 10:17] - alone_b).abs().max() < 1e-5
    assert (cache[0, 10:17] - alone_b).abs().max() > 1e-2


def test_masked_loss_matches_hf_loss_on_padded_batch():
    model = tiny_llama()
    exs = [ChatExample(list(range(3, 3 + n)), [IGNORE_INDEX] * 3 + list(range(6, 3 + n)), ["x"] * n) for n in (9, 14)]
    batch = pad_batch(exs, pad_id=0)
    with torch.no_grad():
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    torch.testing.assert_close(masked_lm_loss(out.logits, batch["labels"]), out.loss, atol=1e-5, rtol=1e-5)
