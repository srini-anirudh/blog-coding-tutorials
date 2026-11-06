import random

import torch

from groundpref.model import TinyVLM, pad, response_mask, sample, seq_logprob, token_logprobs
from groundpref.world import EOS, PAD, PROMPT_LEN, encode, hallucinate, render, sample_scene


def tiny():
    torch.manual_seed(0)
    return TinyVLM(d=16, layers=1, heads=2)


def test_response_mask_skips_prompt_and_padding():
    ids = pad([[1, 4, 9, 10, 2], [1, 4, 2]])
    m = response_mask(ids)
    assert m.tolist() == [[False, True, True, True], [False, True, False, False]]
    assert PROMPT_LEN == 2 and ids[1, -1] == PAD


def test_padding_does_not_change_sequence_logprob():
    m, s = tiny(), sample_scene(random.Random(0))
    img = render(s, torch.Generator().manual_seed(0))[None]
    short = encode(s.claims())
    a = seq_logprob(m, img, torch.tensor([short]))
    b = seq_logprob(m, img.expand(2, -1, -1), pad([short, short + [0, 0, 0] + [EOS]]))
    assert torch.allclose(a, b[:1], atol=1e-5)


def test_shared_prefix_gradients_cancel_exactly():
    """grad log p(y_w,t) - grad log p(y_l,t) is exactly zero before the edit, and not zero at it."""
    m, rng = tiny(), random.Random(1)
    s = sample_scene(rng)
    while len(s.facts) < 2:
        s = sample_scene(rng)
    good = s.claims()
    bad = hallucinate(s, good, "colour", rng)
    w, lo = encode(good), encode(bad)
    edit = next(i for i in range(len(w)) if w[i] != lo[i])
    img = render(s, torch.Generator().manual_seed(0))[None]
    lw = token_logprobs(m, img, torch.tensor([w]))[0]
    ll = token_logprobs(m, img, torch.tensor([lo]))[0]
    params = list(m.parameters())

    def g(x):
        return torch.cat([(t if t is not None else torch.zeros_like(p)).flatten() for t, p in
                          zip(torch.autograd.grad(x, params, retain_graph=True, allow_unused=True), params)])

    for t in range(PROMPT_LEN - 1, edit - 1):
        assert torch.equal(g(lw[t]), g(ll[t]))
    assert (g(lw[edit - 1]) - g(ll[edit - 1])).norm() > 0


def test_uniform_dense_weights_give_the_mean_logprob():
    m, s = tiny(), sample_scene(random.Random(2))
    img = render(s, torch.Generator().manual_seed(0))[None]
    ids = torch.tensor([encode(s.claims())])
    w = torch.ones(1, ids.shape[1] - 1)
    assert torch.allclose(seq_logprob(m, img, ids, w), seq_logprob(m, img, ids) / response_mask(ids).sum())


def test_sample_ends_every_sequence_with_eos():
    m = tiny()
    imgs = torch.randn(3, 8, 8)
    for ids in sample(m, imgs, max_new=5, gen=torch.Generator().manual_seed(0)):
        assert ids[-1] == EOS and ids[:2] == [1, 4]
