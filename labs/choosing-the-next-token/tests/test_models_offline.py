"""Speculative sampling, the decode loop, cached beam search and the contrastive methods on tiny random models.

The models are built from a config with random weights, so nothing is downloaded.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nexttoken import contrastive, desmoothing, lm, search, speculative
from nexttoken.dists import softmax


def tiny(seed: int, layers: int = 2) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(20)  # sharper distributions than the default init
    return m


@pytest.fixture(scope="module")
def models():
    return tiny(0, 1), tiny(1, 2)  # draft, target


def test_toy_speculative_draws_are_distributed_as_the_target():
    g = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(10) * 2, -1)
    q = torch.softmax(torch.randn(10) * 2, -1)
    draws = speculative.toy_speculative_draws(p, q, 200_000, g)
    emp = torch.bincount(draws, minlength=10).float() / 200_000
    assert 0.5 * (emp - p).abs().sum() < 0.01
    assert 0.5 * (q - p).abs().sum() > 0.1  # the draft alone is far from p


def test_decode_with_cache_matches_no_cache(models):
    _, target = models
    prompt = torch.tensor([[1, 5, 9, 2]])
    gen = lm.decode(target, prompt, lambda z, h: z.argmax(-1), 10)
    ids = prompt
    for _ in range(10):
        ids = torch.cat([ids, lm.next_logits(target, ids).argmax(-1, keepdim=True)], 1)
    assert torch.equal(gen, ids[:, 4:])


def test_greedy_speculative_equals_target_greedy(models):
    draft, target = models
    prompt = torch.tensor([[3, 7, 11]])
    spec, stats = speculative.speculative_generate(draft, target, prompt, 20, gamma=3, temperature=0)
    ref = lm.decode(target, prompt, lambda z, h: z.argmax(-1), 20)[0].tolist()
    assert spec == ref
    assert stats["target_calls"] <= 20


def test_speculative_first_token_marginal_is_the_target(models):
    draft, target = models
    g = torch.Generator().manual_seed(0)
    prefix = torch.tensor([[4, 8, 15, 16]]).expand(3000, -1)
    emitted, _ = speculative.speculative_round(draft, target, prefix, gamma=2, generator=g)
    first = torch.tensor([e[0] for e in emitted])
    p = softmax(lm.next_logits(target, prefix[:1]))[0]
    emp = torch.bincount(first, minlength=64).float() / 3000
    iid = torch.bincount(torch.multinomial(p, 3000, replacement=True, generator=g), minlength=64).float() / 3000
    tv, tv_iid = 0.5 * (emp - p).abs().sum(), 0.5 * (iid - p).abs().sum()  # tv_iid: pure sampling noise
    assert tv < 3 * tv_iid + 0.01


def test_cached_beam_search_matches_uncached(models):
    _, target = models
    prompt = torch.tensor([[2, 4, 6]])

    def uncached(prefixes, parents=None):
        ids = torch.cat([prompt.expand(len(prefixes), -1), torch.tensor(prefixes, dtype=torch.long).view(len(prefixes), -1)], 1)
        return torch.log_softmax(lm.next_logits(target, ids), -1)

    a = search.beam_search(search.HFStepper(target, prompt), 0, 3, 8)
    b = search.beam_search(uncached, 0, 3, 8)
    assert [h.tokens for h in a[:3]] == [h.tokens for h in b[:3]]


def test_contrastive_methods_run_and_differ(models):
    amateur, expert = models
    prompt = torch.tensor([[1, 2, 3]])
    cdec = contrastive.contrastive_decode(expert, amateur, prompt, 8)
    greedy = lm.decode(expert, prompt, lambda z, h: z.argmax(-1), 8)
    assert cdec.shape == (1, 8) and greedy.shape == (1, 8)
    cs = contrastive.contrastive_search(expert, prompt, 8, k=4, penalty=0.0)
    assert torch.equal(cs, greedy)  # with no degeneration penalty, contrastive search is greedy
    score = contrastive.contrastive_scores(torch.tensor([[5.0, 4.0, -9.0]]), torch.tensor([[5.0, 1.0, -20.0]]))
    assert score[0, 2] == float("-inf") and score[0, 1] > score[0, 0]


def test_bottleneck_model_never_reaches_zero():
    p_true = desmoothing.make_true_distributions(40, 200, 30, seed=0)
    model = desmoothing.train(p_true, d=8, samples_per_context=50, steps=300)
    p_model = torch.softmax(model(torch.arange(40)), -1).detach()
    rep = desmoothing.support_report(p_model, p_true, {"eta": 3e-3, "top_k": 5})
    assert rep["min_model_prob"] > 0 and rep["leaked_mass_mean"] > 1e-3
    assert 0 < rep["rules"]["eta"]["precision"] <= 1
