"""The thinking controls, tested against a scripted fake model so every token is predictable."""

import re

import pytest

from thinktok.budget import (
    QWEN3_STOP_THINKING, S1_WAIT, ThinkTokens, chat_prompt, think_with_budget, think_with_waits, thought_was_empty,
)

SPECIAL = {"<think>": 1000, "</think>": 1001, "<|im_end|>": 1002}
INV = {v: k for k, v in SPECIAL.items()}
_SPLIT = re.compile("(" + "|".join(map(re.escape, SPECIAL)) + ")")


class CharTok:
    """One id per character, plus three special tokens."""

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in _SPLIT.split(text):
            ids += [SPECIAL[piece]] if piece in SPECIAL else [ord(c) for c in piece]
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join("" if (i in INV and skip_special_tokens) else INV.get(i, chr(i) if i < 1000 else "") for i in ids)


TOK = CharTok()
TT = ThinkTokens(think_end=SPECIAL["</think>"], eos=SPECIAL["<|im_end|>"])


def fake_model(think_len):
    """Thinks with 't' tokens until it has written think_len of them since the last 'Wait' (or the start),
    then closes </think>. Once </think> is in the context it answers '42' and ends the turn."""
    calls = []

    def generate(prompts, max_new, stop_ids):
        calls.append(max_new)
        outs = []
        for p in prompts:
            if SPECIAL["</think>"] in p:
                seq = [ord("4"), ord("2"), TT.eos]
            else:
                text = TOK.decode(p)
                since = text.rsplit(S1_WAIT, 1)[-1].count("t")
                seq = [ord("t")] * max(0, think_len - since) + [TT.think_end]
            out = []
            for t in seq[:max_new]:
                out.append(t)
                if t in stop_ids:
                    break
            outs.append(out)
        return outs

    generate.calls = calls
    return generate


PROMPT = [TOK.encode("Q?")]


def test_budget_zero_is_the_hard_switch():
    (tr,) = think_with_budget(fake_model(10), PROMPT, TOK, TT, budget=0)
    assert tr.thinking == []
    assert TOK.decode(tr.injected) == "<think>\n\n</think>\n\n"
    assert TOK.decode(tr.answer, skip_special_tokens=True) == "42"


def test_budget_not_reached_leaves_the_trace_alone():
    (tr,) = think_with_budget(fake_model(10), PROMPT, TOK, TT, budget=50)
    assert not tr.truncated and tr.injected == []
    assert len(tr.thinking) == 11 and tr.thinking[-1] == TT.think_end


def test_budget_reached_interrupts_and_salvages():
    (tr,) = think_with_budget(fake_model(100), PROMPT, TOK, TT, budget=30)
    assert tr.truncated and len(tr.thinking) == 30
    assert TOK.decode(tr.injected) == QWEN3_STOP_THINKING  # spliced in verbatim, </think> included
    assert TOK.decode(tr.answer, skip_special_tokens=True) == "42"  # the model still answers
    assert tr.stream == tr.thinking + tr.injected + tr.answer


def test_waits_replace_the_end_of_thinking():
    (tr,) = think_with_waits(fake_model(5), PROMPT, TOK, TT, waits=2, cap=1000)
    assert tr.waits == 2 and not tr.truncated
    text = TOK.decode(tr.stream)
    assert text.count(S1_WAIT) == 2 and text.count("</think>") == 1
    assert tr.thinking.count(ord("t")) == 15  # three rounds of five
    assert TOK.decode(tr.answer, skip_special_tokens=True) == "42"


def test_waits_respect_the_cap():
    trs = think_with_waits(fake_model(8), PROMPT * 2, TOK, TT, waits=5, cap=20)
    for tr in trs:
        assert tr.truncated and len(tr.thinking) == 20
        assert TOK.decode(tr.injected).endswith("\n</think>\n\n")


def test_zero_waits_is_plain_thinking():
    (tr,) = think_with_waits(fake_model(7), PROMPT, TOK, TT, waits=0, cap=100)
    assert tr.waits == 0 and tr.injected == [] and len(tr.thinking) == 8


def test_thought_was_empty():
    assert thought_was_empty("<think>\n\n</think>\n\nThe answer is 3.")
    assert not thought_was_empty("<think>\nLet me add 2 and 3")
    assert thought_was_empty("The answer is 3.")


def _cached_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3 tokenizer not in the local cache")


def test_chat_prompt_modes_from_cache():
    tok = _cached_tokenizer()
    hard = tok.decode(chat_prompt(tok, "2+3?", "hard_no_think"))
    soft = tok.decode(chat_prompt(tok, "2+3?", "soft_no_think"))
    think = tok.decode(chat_prompt(tok, "2+3?", "think"))
    assert hard.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "2+3? /no_think<|im_end|>" in soft and soft.endswith("assistant\n")
    assert think.endswith("assistant\n")
    with pytest.raises(ValueError):
        chat_prompt(tok, "x", "maybe")


@pytest.mark.network
def test_qwen3_vocab_and_think_tokens():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    assert len(tok) == 151_669
    assert len(tok.encode("</think>", add_special_tokens=False)) == 1


@pytest.mark.gpu
@pytest.mark.network
def test_budget_on_real_qwen3():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from thinktok.hf import load, make_generate

    model, tok = load("Qwen/Qwen3-0.6B", device="cuda")
    toks = ThinkTokens(think_end=tok.convert_tokens_to_ids("</think>"), eos=tok.convert_tokens_to_ids("<|im_end|>"))
    prompt = chat_prompt(tok, "What is 17 * 23? Put the answer in \\boxed{}.", "think")
    (tr,) = think_with_budget(make_generate(model, tok), [prompt], tok, toks, budget=8, max_answer=64)
    assert tr.truncated and len(tr.thinking) == 8
    assert tr.answer  # the model answered after the interruption
