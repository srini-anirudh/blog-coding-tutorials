import math

import pytest
import torch

from pixelcost import connectors as cn
from pixelcost import selection as sel


def test_pixel_shuffle_is_lossless_and_pooling_is_not():
    x = torch.randn(3, 64, 8)
    assert torch.equal(cn.unshuffle(cn.pixel_shuffle(x, 2), 2), x)
    assert cn.avg_pool(x, 2).shape == (3, 16, 8)
    g = torch.Generator().manual_seed(0)
    xs, ys = cn.make_task("glyph", 64, g)
    pooled = cn.avg_pool(xs, 2)
    by_label = [pooled[ys == k, :, 0].mean(0) for k in range(4)]   # stroke channel after pooling
    assert all(torch.allclose(by_label[0][6], b[6], atol=0.3) for b in by_label)  # the glyph block: same mean


@pytest.mark.parametrize("kind,tokens", [("mlp", 64), ("shuffle", 16), ("pool", 16), ("resampler", 4), ("conv", 16)])
def test_connector_token_counts(kind, tokens):
    assert cn.Connector(kind)(torch.randn(2, 64, 8)).shape == (2, tokens, 32)


@pytest.mark.slow
def test_pool_fails_the_glyph_task_and_shuffle_solves_it():
    assert cn.train_and_eval("pool", "glyph", 300)["test_accuracy"] < 0.4
    assert cn.train_and_eval("shuffle", "glyph", 300)["test_accuracy"] > 0.9


@pytest.mark.parametrize("name", list(sel.POLICIES))
@pytest.mark.parametrize("k", [144, 64])
def test_every_policy_returns_k_tokens_in_raster_order(name, k):
    feats = torch.randn(576, 16)
    red = sel.POLICIES[name](feats, 24, 24, k, scores=torch.rand(576), seed=1)
    assert red.features.shape == (k, 16)
    assert len(set(red.index.tolist())) == k and int(red.index.max()) < 576
    assert torch.equal(red.index, red.index.sort().values)
    if name in ("pool", "tome"):
        assert int(red.size.sum()) == 576               # every original token is accounted for


def test_attention_rank_keeps_the_top_scores():
    scores = torch.arange(100.0)
    red = sel.attention_rank(torch.randn(100, 4), 10, 10, 5, scores=scores)
    assert red.index.tolist() == [95, 96, 97, 98, 99]


def test_pool_is_block_mean_at_block_centre():
    feats = torch.randn(36, 3)
    red = sel.avg_pool(feats, 6, 6, 4)
    grid = feats.reshape(6, 6, 3)
    assert torch.allclose(red.features[0], grid[:3, :3].reshape(-1, 3).mean(0), atol=1e-6)
    assert red.index.tolist() == [1 * 6 + 1, 1 * 6 + 4, 4 * 6 + 1, 4 * 6 + 4]


def test_diversity_never_keeps_duplicates_while_distinct_tokens_remain():
    base = torch.randn(8, 16)
    feats = torch.cat([base, base])                    # every token has an exact twin
    red = sel.diversity(feats, 4, 4, 8)
    assert len({tuple(f.tolist()) for f in red.features}) == 8


def test_tome_merge_preserves_size_weighted_mean():
    x = torch.randn(24, 8)
    red = sel.tome(x, 4, 6, 7)
    assert len(red.features) == 7
    assert torch.allclose((red.features * red.size[:, None]).sum(0), x.sum(0), atol=1e-4)


def test_tome_merges_identical_tokens_first():
    x = torch.randn(8, 4)
    x[2] = x[3]                                         # an A/B pair that is identical
    merged, size, index = sel.bipartite_merge(x, torch.ones(8, dtype=torch.long), torch.arange(8), r=1)
    assert 2 not in index.tolist() and size[index.tolist().index(3)] == 2


def test_proportional_attention_equals_attention_over_the_unmerged_copies():
    q, k, v = torch.randn(3, 8), torch.randn(4, 8), torch.randn(4, 5)
    dup_k, dup_v = torch.cat([k, k[:1], k[:1]]), torch.cat([v, v[:1], v[:1]])  # token 0 seen three times
    plain = (q @ dup_k.T / math.sqrt(8)).softmax(-1) @ dup_v
    merged = sel.proportional_attention(q, k, v, torch.tensor([3, 1, 1, 1]))
    assert torch.allclose(plain, merged, atol=1e-5)
