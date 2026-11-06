"""Tokenizers, token budgets, token priors and guidance."""

import math

import pytest
import torch
import torch.nn.functional as F

from seedraw import data
from seedraw.priors import TokenPrior, cfg_logits, entropy_bits, train_prior
from seedraw.tokenizers import FSQ, Tokenizer, VectorQuantizer, codebook_stats, ste, token_budget, train_tokenizer


def test_token_budget_matches_the_post():
    cham = token_budget(512, 512, 16, 8192)
    assert cham["tokens"] == 1024 and cham["max_bits"] == 13_312
    assert round(cham["compression"]) == 473
    emu3 = token_budget(512, 512, 8, 32768)
    assert emu3["tokens"] == 4096 and emu3["max_bits"] == 61_440
    assert token_budget(384, 384, 16, 16384)["tokens"] == 576
    assert math.isclose(math.log2(150_000), 17.19, abs_tol=0.01)


def test_straight_through_copies_the_gradient():
    x = torch.randn(5, requires_grad=True)
    y = ste(x, x.round())
    assert torch.equal(y, x.detach().round())
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones(5))


def test_vq_picks_the_nearest_code():
    vq = VectorQuantizer(4, 2)
    with torch.no_grad():
        vq.embed.weight.copy_(torch.tensor([[0.0, 0], [1, 0], [0, 1], [5, 5]]))
    z = torch.tensor([[[0.9, 0.1], [4.0, 4.5], [0.1, 0.2]]])
    zq, idx, loss = vq(z)
    assert idx.tolist() == [[1, 3, 0]]
    assert torch.allclose(zq, vq.embed.weight[idx])
    assert loss > 0


@pytest.mark.parametrize("quantizer", [
    lambda: VectorQuantizer(64, 64), lambda: VectorQuantizer(64, 64, code_dim=8, normalize=True),
    lambda: FSQ([4, 4, 4], 64), lambda: FSQ([8, 5, 5, 5], 64),
])
def test_decode_indices_matches_the_forward_pass(quantizer):
    tok = Tokenizer(quantizer())
    x = data.shapes(8, seed=2)["images"]
    recon, idx, _ = tok(x)
    assert idx.shape == (8, 16)
    assert 0 <= idx.min() and idx.max() < tok.quantizer.codebook
    assert torch.allclose(tok.decode_indices(idx), recon, atol=1e-5)


def test_fsq_reaches_every_code():
    fsq = FSQ([3, 4], 2)
    with torch.no_grad():
        fsq.down.weight.copy_(torch.eye(2) * 10)
        fsq.down.bias.zero_()
    grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 41), torch.linspace(-1, 1, 41), indexing="ij"), -1)
    _, idx, _ = fsq(grid.view(1, -1, 2))
    assert sorted(idx.unique().tolist()) == list(range(12))


def test_codebook_stats():
    s = codebook_stats(torch.tensor([0, 0, 1, 1]), 4)
    assert s["usage"] == 0.5 and math.isclose(s["perplexity"], 2.0, rel_tol=1e-6)


def test_cfg_scale_one_is_conditional_and_entropy_falls():
    lu, lc = torch.randn(12), torch.randn(12)
    assert torch.allclose(cfg_logits(lu, lc, 1.0), lc)
    assert torch.allclose(cfg_logits(lu, lc, 0.0), lu)
    h = [float(entropy_bits(cfg_logits(lu, lc, s))) for s in (1, 3, 10)]
    assert h[0] > h[1] > h[2]
    # Bayes form: guided log-probs equal log p(x) + s (log p(x|c) - log p(x)) up to a constant
    s = 3.0
    g = F.log_softmax(cfg_logits(lu, lc, s), -1)
    b = F.log_softmax(lu, -1) + s * (F.log_softmax(lc, -1) - F.log_softmax(lu, -1))
    assert torch.allclose(g, F.log_softmax(b, -1), atol=1e-5)


def test_raster_prior_is_causal_and_needs_one_pass_per_token():
    torch.manual_seed(0)
    prior = TokenPrior(10, 16, 4, "raster", d=32, layers=1).eval()
    toks = torch.randint(0, 10, (2, 16))
    later = toks.clone()
    later[:, 9:] = (later[:, 9:] + 1) % 10
    a = prior.logits(toks, torch.tensor([0, 1]))
    b = prior.logits(later, torch.tensor([0, 1]))
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)  # logits up to position 9 only see tokens < 9
    assert not torch.allclose(a[:, 10:], b[:, 10:])
    prior.passes = 0
    prior.sample_raster(torch.tensor([0, 1, 2]))
    assert prior.passes == 16


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_maskgit_fills_the_grid_in_its_round_count(steps):
    prior = TokenPrior(10, 16, 4, "maskgit", d=32, layers=1).eval()
    prior.passes = 0
    out = prior.sample_maskgit(torch.tensor([0, 3]), steps=steps)
    assert prior.passes == steps
    assert (out != prior.mask_id).all() and out.max() < 10


def test_guided_sampling_doubles_the_batch_not_the_passes():
    prior = TokenPrior(10, 16, 4, "raster", d=32, layers=1).eval()
    prior.passes = 0
    prior.sample_raster(torch.tensor([0, 1]), scale=3.0)
    assert prior.passes == 16


def test_prior_learns_a_class_dependent_token():
    torch.manual_seed(0)
    cond = torch.randint(0, 4, (512,))
    toks = torch.randint(0, 10, (512, 16))
    toks[:, 0] = cond  # the first token encodes the class
    prior = TokenPrior(10, 16, 4, "raster", d=32, layers=1)
    train_prior(prior, toks, cond, steps=150, batch=128, lr=3e-3)
    prior.eval()
    out = prior.sample_raster(torch.arange(4).repeat(16))
    assert (out[:, 0] == torch.arange(4).repeat(16)).float().mean() > 0.9


@pytest.mark.slow
def test_fsq_uses_more_of_its_codebook_than_plain_vq():
    imgs = data.shapes(2000, seed=0)["images"]
    usage = {}
    for name, q in {"vq": VectorQuantizer(256, 64), "fsq": FSQ([4, 4, 4, 4], 64)}.items():
        torch.manual_seed(0)
        tok = Tokenizer(q)
        train_tokenizer(tok, imgs, steps=300)
        with torch.no_grad():
            usage[name] = codebook_stats(tok.encode(imgs)[1], q.codebook)["usage"]
    assert usage["fsq"] > usage["vq"]
