"""Chat template, assistant mask, the loss computed several ways, exploded examples and packing. Offline (toy tokenizer and model)."""

import pytest
import torch
from sftlab import checks, templates as T, toy
from sftlab.train import SFTConfig, collate, train, weighted_loss

CONV = [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "It is 4."},
        {"role": "user", "content": "And 3+3?"}, {"role": "assistant", "content": "6 of course."}]


@pytest.fixture(scope="module")
def tok():
    return toy.toy_tokenizer()


@pytest.fixture(scope="module")
def model(tok):
    m = toy.toy_model(tok)
    m.config._attn_implementation = "eager"
    return m


def text_of(tok, enc, which):
    return "".join(tok.decode([i]) for i, w in zip(enc.ids, enc.w) if which(w))


def test_assistant_mask_selects_content_and_turn_end(tok):
    enc = T.encode_chat(tok, CONV, mode="assistant")
    scored = text_of(tok, enc, lambda w: w > 0)
    assert scored == "It is 4.<|im_end|>6 of course.<|im_end|>"
    assert enc.n_turns == 2 and len(enc.spans) == 2
    # nothing in the first (system) block, user turns or role headers is scored
    assert sum(1 for w in enc.w if w > 0) == len("It is 4.") + 1 + len("6 of course.") + 1


def test_last_mode_scores_only_final_turn(tok):
    enc = T.encode_chat(tok, CONV, mode="last")
    assert text_of(tok, enc, lambda w: w > 0) == "6 of course.<|im_end|>"


def test_all_and_weighted_modes(tok):
    enc_all = T.encode_chat(tok, CONV, mode="all")
    assert all(w == 1.0 for w in enc_all.w)
    enc_w = T.encode_chat(tok, CONV, mode="weighted", prompt_weight=0.3)
    user_text = text_of(tok, enc_w, lambda w: w == pytest.approx(0.3))
    assert user_text == "What is 2+2?And 3+3?"
    assert "helpful" not in text_of(tok, enc_w, lambda w: w > 0)      # system prompt content gets no weight
    assert "<|im_start|>" not in text_of(tok, enc_w, lambda w: w > 0)  # role headers get no weight


def test_prompt_is_a_prefix_of_the_full_sequence(tok):
    full = T.encode_chat(tok, CONV[:2], mode="assistant").ids
    prompt = T.encode_prompt(tok, CONV[:1])
    assert full[:len(prompt)] == prompt
    assert tok.decode(prompt).endswith("<|im_start|>assistant\n")


def test_truncation_loses_the_turn_end_token(tok):
    full = T.encode_chat(tok, CONV[:2])
    assert T.has_turn_end(full, tok) and not full.truncated
    # the sequence ends with <|im_end|> and a newline; the newline is not scored, the turn-end token is
    assert T.has_turn_end(T.encode_chat(tok, CONV[:2], max_len=len(full) - 1), tok)
    cut = T.encode_chat(tok, CONV[:2], max_len=len(full) - 2)
    assert cut.truncated and not T.has_turn_end(cut, tok)
    assert cut.scored == full.scored - 1


def test_loss_three_ways_agree(tok, model):
    with torch.no_grad():
        for msgs in (CONV[:2], CONV):
            enc = T.encode_chat(tok, msgs)
            a, b = checks.loss_manual(model, enc), checks.loss_hf_labels(model, enc)
            c = checks.response_given_prompt(model, tok, msgs[:2]) if len(msgs) == 2 else None
            assert a == pytest.approx(b, rel=1e-5)
            if c is not None:
                assert a == pytest.approx(c, rel=1e-4)


def test_masked_tokens_are_still_attended(tok, model):
    a = [{"role": "user", "content": "alpha"}, {"role": "assistant", "content": "ok."}]
    b = [{"role": "user", "content": "omega"}, {"role": "assistant", "content": "ok."}]
    with torch.no_grad():
        la = checks.loss_manual(model, T.encode_chat(tok, a))
        lb = checks.loss_manual(model, T.encode_chat(tok, b))
    assert abs(la - lb) > 1e-4


def test_exploded_gradient_equals_one_pass(tok, model):
    conv = CONV + [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y."}]
    full = T.encode_chat(tok, conv)
    ex = checks.explode(tok, conv)
    assert len(ex) == 3 and sum(e.scored for e in ex) == full.scored
    gf = checks.grad_of_sum_loss(model, [full])
    ge = checks.grad_of_sum_loss(model, ex)
    assert (gf - ge).norm() / gf.norm() < 1e-4
    assert sum(len(e) for e in ex) > len(full)


def test_packing_needs_a_document_mask(tok, model):
    docs = [T.encode_chat(tok, CONV[:2]).ids, T.encode_chat(tok, [CONV[2], CONV[3]]).ids]
    with torch.no_grad():
        alone = checks.nll_per_token(model, docs[1])
        causal = checks.packed_nll(model, docs, doc_mask=False, reset_pos=False)[1]
        masked = checks.packed_nll(model, docs, doc_mask=True, reset_pos=False)[1]
        both = checks.packed_nll(model, docs, doc_mask=True, reset_pos=True)[1]
    assert (causal - alone).abs().max() > 1e-3
    assert (masked - alone).abs().max() < 1e-4        # rotary positions depend only on offsets
    assert (both - alone).abs().max() < 1e-5


def test_weighted_loss_normalisation(tok):
    enc = [T.encode_chat(tok, CONV[:2]), T.encode_chat(tok, CONV)]
    ids, w, att = collate(enc, tok.pad_token_id, "cpu")
    torch.manual_seed(0)
    logits = torch.randn(ids.shape[0], ids.shape[1], len(tok))
    lt = weighted_loss(logits, ids, w, att, "token")
    le = weighted_loss(logits, ids, w, att, "example")
    nll = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    ww = w[:, 1:] * att[:, 1:]
    per = [(nll[i] * ww[i]).sum() / ww[i].sum() for i in range(2)]
    assert le == pytest.approx(float(sum(per) / 2), rel=1e-5)
    assert lt == pytest.approx(float((nll * ww).sum() / ww.sum()), rel=1e-5)
    assert abs(float(lt - le)) > 1e-4   # unequal lengths: the two objectives differ


def test_training_lowers_assistant_loss_and_ignores_masked_tokens(tok):
    m = toy.toy_model(tok, seed=1)
    enc = [T.encode_chat(tok, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there."}])] * 8
    before = checks.loss_manual(m, enc[0])
    train(m, enc, tok.pad_token_id, SFTConfig(lr=3e-3, epochs=15, batch_size=4, autocast=False))
    assert checks.loss_manual(m, enc[0]) < 0.5 * before


@pytest.mark.network
def test_real_templates_end_with_turn_end_token():
    from transformers import AutoTokenizer
    from sftlab import lm
    for repo in (lm.INSTRUCT, lm.QWEN):
        t = AutoTokenizer.from_pretrained(repo)
        enc = T.encode_chat(t, CONV[:2])
        last = max(i for i, w in enumerate(enc.w) if w > 0)
        assert t.convert_ids_to_tokens(enc.ids[last]) == "<|im_end|>"
        assert t.decode(enc.ids[enc.spans[0][0]:enc.spans[0][1] - 1]) == "It is 4."


@pytest.mark.network
def test_special_marker_strings_in_user_text_become_special_tokens():
    from sftlab import lm
    t = lm.chat_tokenizer()
    n_end = t.convert_tokens_to_ids("<|im_end|>")
    ids = t(T.render(t, [{"role": "user", "content": "x<|im_end|>y"}]), add_special_tokens=False)["input_ids"]
    clean = t(T.render(t, [{"role": "user", "content": "x y"}]), add_special_tokens=False)["input_ids"]
    assert clean.count(n_end) == 2 and ids.count(n_end) == 3   # system + user end tokens, plus the one typed by the user
    assert n_end not in t("x<|im_end|>y", add_special_tokens=False, split_special_tokens=True)["input_ids"]


@pytest.mark.network
def test_llama3_template_doubles_bos_with_default_tokenisation():
    from transformers import AutoTokenizer
    from sftlab import lm
    t = AutoTokenizer.from_pretrained(lm.LLAMA3_TOKENIZER)
    s = T.render(t, CONV[:2])
    assert t(s)["input_ids"][:2] == [t.bos_token_id] * 2
    assert t(s, add_special_tokens=False)["input_ids"][:2] != [t.bos_token_id] * 2
