"""The decoding loops and the arithmetic task. A scripted fake model keeps these offline and exact."""

from types import SimpleNamespace

import pytest
import torch

from ttc.arith import LEVELS, last_digit_check, make_problems, parse_answer, shape_check
from ttc.lm import Forcing, budget_force, sample

PAD, EOS, STOP, NL, A, B, WAIT, DIGIT, COLON = range(9)
TS = {"stop": {STOP, EOS}, "end_answer": {NL, EOS}, "eos": {EOS}, "answer": [NL, COLON], "pad": PAD}


class Fake(torch.nn.Module):
    """Thinks with token A for 3 tokens, then tries to stop. After 'Answer:' it says one digit and a newline.

    The 'cache' is just the number of tokens seen, so position bookkeeping is tested too.
    """

    def __init__(self, prompt_len):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.prompt_len = prompt_len

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True):
        seen = (past_key_values or 0) + input_ids.shape[1]
        assert attention_mask.shape[1] == seen
        assert int(position_ids[0, -1]) == seen - 1
        last = input_ids[:, -1].tolist()
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 9)
        for r, t in enumerate(last):
            if t == COLON:
                y = DIGIT
            elif t == DIGIT:
                y = NL
            elif seen - self.prompt_len < 3:
                y = A
            else:
                y = STOP
            logits[r, -1, y] = 50.0
        return SimpleNamespace(logits=logits, past_key_values=seen)


PROMPT = [[B, B], [B, B]]


def test_natural_stop():
    rows = budget_force(Fake(2), PROMPT, Forcing(), TS)
    assert rows[0]["think"] == [A, A, A] and rows[0]["answer"] == [DIGIT, NL]
    assert rows[0]["stops_seen"] == 1 and not rows[0]["capped"]


def test_wait_suppresses_stop_and_extends():
    rows = budget_force(Fake(2), PROMPT, Forcing(waits=2, wait_ids=[WAIT]), TS)
    assert rows[1]["think"] == [A, A, A, WAIT, WAIT]
    assert rows[1]["waits_used"] == 2 and rows[1]["stops_seen"] == 3
    assert rows[1]["answer"] == [DIGIT, NL]


def test_cap_forces_answer():
    rows = budget_force(Fake(2), PROMPT, Forcing(cap=2), TS)
    assert rows[0]["think"] == [A, A] and rows[0]["capped"] and rows[0]["answer"] == [DIGIT, NL]


def test_max_think_stops_runaway():
    rows = budget_force(Fake(2), PROMPT, Forcing(waits=100, wait_ids=[WAIT], max_think=6), TS)
    assert len(rows[0]["think"]) == 6 and rows[0]["capped"]


def test_sample_stops_at_eos_and_reports_logprob():
    toks, lp = sample(Fake(2), PROMPT, eos={STOP}, pad=PAD, max_new=10, temperature=1.0)
    assert toks == [[A, A, A], [A, A, A]]
    assert lp[0] == pytest.approx(0.0, abs=1e-6)  # the fake model is certain


def test_arith_problems_and_checks():
    probs = make_problems(per_level=3)
    assert len(probs) == 3 * len(LEVELS)
    assert all(p["truth"] == eval(p["expr"]) for p in probs)
    assert make_problems(per_level=3) == probs  # seeded
    assert parse_answer("so 12 then Answer: 1,234") == 1234
    assert parse_answer(r"result \boxed{56}.") == 56
    assert parse_answer("no digits") is None
    assert last_digit_check(1004, 2114) and not shape_check(1004, 21114) and shape_check(1004, 2114)
    assert not last_digit_check(None, 3)


@pytest.mark.network
def test_loops_run_on_a_real_tiny_model():
    from ttc.lm import chat_ids, load, token_sets

    model, tok = load("hf-internal-testing/tiny-random-gpt2", "cpu")
    tok.pad_token = tok.eos_token
    ids = [tok.encode("1 + 1 ="), tok.encode("What is 2 * 3? Think.")]
    toks, lp = sample(model, ids, eos={tok.eos_token_id}, pad=tok.eos_token_id, max_new=8, temperature=1.0)
    assert len(toks) == 2 and all(len(t) <= 8 for t in toks) and all(x <= 0 for x in lp)
    ts = token_sets(tok)
    rows = budget_force(model, ids, Forcing(cap=5), ts)
    assert all(len(r["think"]) == 5 and r["capped"] for r in rows)
