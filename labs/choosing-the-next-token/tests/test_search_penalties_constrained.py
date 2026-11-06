"""Greedy and beam search on the toy tree, the penalties against Hugging Face's processors, and the automaton."""

import math

import torch
from transformers.generation.logits_process import (NoRepeatNGramLogitsProcessor,
                                                    RepetitionPenaltyLogitsProcessor)

from nexttoken import constrained as cd
from nexttoken import penalties as pen
from nexttoken import search


def test_greedy_is_locally_right_and_globally_wrong():
    lm = search.post_example_tree()
    g = search.greedy(lm, lm.eos_id, 5)
    best = search.exhaustive(lm, 5)[0]
    assert [lm.vocab[i] for i in g.tokens] == ["A", "x", "<eos>"]
    assert math.isclose(math.exp(g.logprob), 0.21, rel_tol=1e-6)
    assert [lm.vocab[i] for i in best.tokens] == ["B", "x", "<eos>"]
    assert math.isclose(math.exp(best.logprob), 0.36, rel_tol=1e-6)


def test_beam_width_one_is_greedy_and_width_two_finds_the_optimum():
    lm = search.post_example_tree()
    assert search.beam_search(lm, lm.eos_id, 1, 5)[0].tokens == search.greedy(lm, lm.eos_id, 5).tokens
    assert search.beam_search(lm, lm.eos_id, 2, 5)[0].tokens == search.exhaustive(lm, 5)[0].tokens


def test_length_penalty_changes_only_the_ranking():
    vocab = ["a", "b", "<eos>"]
    table = {(): {"<eos>": 0.4, "a": 0.6}, ("a",): {"b": 0.6, "<eos>": 0.4}, ("a", "b"): {"<eos>": 1.0}}
    lm = search.ToyTreeLM(vocab, table)
    raw = search.beam_search(lm, lm.eos_id, 3, 5, alpha=0.0)
    norm = search.beam_search(lm, lm.eos_id, 3, 5, alpha=1.0)
    assert raw[0].tokens == [2]  # the empty output wins on raw log-probability
    assert len(norm[0].tokens) > 1
    assert sorted(map(tuple, (h.tokens for h in raw))) == sorted(map(tuple, (h.tokens for h in norm)))


def test_repetition_penalty_matches_hf_and_naive_has_sign_trap():
    torch.manual_seed(0)
    z = torch.randn(2, 30) * 3
    hist = torch.randint(0, 30, (2, 6))
    ours = pen.repetition_penalty(z, hist, 1.3)
    ref = RepetitionPenaltyLogitsProcessor(1.3)(hist, z.clone())
    assert torch.allclose(ours, ref)
    neg = torch.tensor([[2.0, -1.0, -3.0]])
    h = torch.tensor([[1]])
    p_before = torch.softmax(neg, -1)[0, 1]
    assert torch.softmax(pen.repetition_penalty_naive(neg, h, 2.0), -1)[0, 1] > p_before  # the trap
    assert torch.softmax(pen.repetition_penalty(neg, h, 2.0), -1)[0, 1] < p_before


def test_repetition_penalty_is_not_shift_invariant_but_frequency_penalty_is():
    z = torch.tensor([[2.0, 1.0, 0.5, -0.5]])
    h = torch.tensor([[0, 2]])
    a = torch.softmax(pen.repetition_penalty(z, h, 1.5), -1)
    b = torch.softmax(pen.repetition_penalty(z + 5.0, h, 1.5), -1)
    assert not torch.allclose(a, b, atol=1e-3)
    c = torch.softmax(pen.frequency_penalty(z, h, 0.7), -1)
    d = torch.softmax(pen.frequency_penalty(z + 5.0, h, 0.7), -1)
    assert torch.allclose(c, d, atol=1e-6)


def test_no_repeat_ngram_matches_hf():
    torch.manual_seed(1)
    z = torch.randn(3, 12)
    hist = torch.randint(0, 4, (3, 15))
    for n in (2, 3):
        assert torch.equal(pen.no_repeat_ngram(z, hist, n), NoRepeatNGramLogitsProcessor(n)(hist, z.clone()))


def test_presence_pays_once_frequency_scales():
    z = torch.zeros(1, 4)
    h = torch.tensor([[1, 1, 1, 2]])
    assert pen.presence_penalty(z, h, 1.0).tolist() == [[0.0, -1.0, -1.0, 0.0]]
    assert pen.frequency_penalty(z, h, 1.0).tolist() == [[0.0, -3.0, -1.0, 0.0]]


def test_dfa_accepts_and_rejects():
    dfa = cd.CharDFA(cd.PERSON_PATTERN)
    assert dfa.matches('{"name": "Alice Smith", "age": 31}')
    assert not dfa.matches('{"name": "Alice", "age": }')
    assert not dfa.matches('{"name": "", "age": 3}')
    assert not dfa.matches('{"name": "Al1ce", "age": 3}')
    assert not dfa.matches('{"name": "A", "age": 1234}')


def test_token_index_on_a_fake_vocabulary():
    vocab = ['{"', 'name', '":', ' "', 'Bob', '", "', 'age', '": ', '4', '2', '}', 'x1', '<eos>']
    dfa = cd.CharDFA(cd.PERSON_PATTERN)
    idx = cd.TokenIndex(dfa, vocab, eos_id=12)
    state = dfa.start
    for piece in ['{"', 'name', '":', ' "', 'Bob', '", "', 'age', '": ', '4', '2', '}']:
        allowed = idx.allowed(state)
        assert vocab.index(piece) in allowed and 11 not in allowed and 12 not in allowed
        state = allowed[vocab.index(piece)]
    assert state == dfa.accept and set(idx.allowed(state)) == {12}


def test_masking_is_not_conditioning_and_importance_resampling_fixes_it():
    V, L = 3, 4
    table = cd.toy_lm(V, L, seed=0)

    def in_lang(y):
        return y[-1] == 2 and all(not (a == 0 and b == 0) for a, b in zip(y, y[1:]))

    exact = cd.exact_conditional(table, V, L, in_lang)
    masked = cd.masked_distribution(table, V, L, in_lang)
    assert math.isclose(sum(masked.values()), 1.0, rel_tol=1e-6)
    tv_mask = cd.total_variation(masked, exact)
    assert tv_mask > 0.05
    fixed = cd.importance_resample(table, V, L, in_lang, n_particles=64, n_rounds=20000)
    assert cd.total_variation(fixed, exact) < tv_mask / 3
