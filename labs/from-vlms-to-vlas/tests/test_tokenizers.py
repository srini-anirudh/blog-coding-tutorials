import math

import numpy as np
import pytest

from vlalab.tokenizers import (ActionTokenCodec, BytePairEncoder, FASTTokenizer, UniformBinner,
                               conditional_entropy_bits, dct_matrix, least_used_token_ids, smooth_actions)


def test_bins_round_trip_within_half_a_bin():
    acts = smooth_actions(50, 2.0, 20, seed=0)
    b = UniformBinner(256).fit(acts)
    x = np.clip(acts, b.low, b.high)
    err = np.abs(b.decode(b.encode(x)) - x) / (b.high - b.low)
    assert err.max() <= 1 / 512 + 1e-9
    assert b.encode(acts).min() >= 0 and b.encode(acts).max() <= 255


def test_bins_use_percentiles_so_outliers_are_clipped():
    a = np.zeros((1000, 1))
    a[:, 0] = np.linspace(-1, 1, 1000)
    a[0, 0] = -100.0
    b = UniformBinner(256).fit(a)
    assert b.low[0] > -1.0  # the outlier does not set the range
    assert b.encode(np.array([[-100.0]]))[0, 0] == 0


def test_least_used_ids_skip_specials_and_pick_rare_ids():
    counts = {i: 100 for i in range(10)}
    counts.update({3: 0, 7: 1})
    ids = least_used_token_ids(counts, 12, k=3, protected={11})
    assert ids == sorted([3, 10, 7])  # 10 never occurs, 11 is protected


def test_codec_round_trip_and_mask():
    acts = smooth_actions(20, 1.0, 10, seed=1)
    b = UniformBinner(256).fit(acts)
    codec = ActionTokenCodec(b, list(range(1000, 1256)))
    a = np.clip(acts[0, 0], b.low, b.high)
    ids = codec.encode(a)
    assert len(ids) == 7 and all(1000 <= i < 1256 for i in ids)
    assert np.allclose(codec.decode(ids), a, atol=(b.high - b.low).max() / 256)
    mask = codec.logit_mask(2000)
    assert np.isinf(mask[:1000]).all() and (mask[1000:1256] == 0).all()


def test_dct_is_orthonormal_and_matches_the_post_formula():
    n = 16
    m = dct_matrix(n)
    assert np.allclose(m @ m.T, np.eye(n), atol=1e-12)
    a = np.random.default_rng(0).normal(size=n)
    post = np.array([sum(a[i] * math.cos(math.pi * k * (2 * i + 1) / (2 * n)) for i in range(n)) for k in range(n)])
    scale = np.full(n, math.sqrt(2 / n))
    scale[0] = math.sqrt(1 / n)
    assert np.allclose(m @ a, post * scale)


def test_bpe_round_trip_and_compression():
    seqs = [[1, 2, 0, 0, 0, 0, 3] * 3 for _ in range(20)]
    bpe = BytePairEncoder(vocab_size=20).fit(seqs, alphabet=list(range(5)))
    enc = bpe.encode(seqs[0])
    assert bpe.decode(enc) == seqs[0]
    assert len(enc) < len(seqs[0])
    assert bpe.decode(bpe.encode([4, 4, 1])) == [4, 4, 1]  # symbols unseen in training still encode


def test_fast_round_trip_and_fewer_tokens_than_bins():
    train = smooth_actions(60, 1.0, 50, seed=3)
    fast = FASTTokenizer(scale=10.0, vocab_size=600).fit(train)  # base alphabet: 201 symbols
    test = smooth_actions(5, 1.0, 50, seed=4)
    for c in test:
        ids = fast.encode(c)
        assert len(ids) < 350
        rec = fast.decode(ids)
        clipped = np.clip(c, fast.binner.low, fast.binner.high)
        rng = fast.binner.high - fast.binner.low
        assert np.sqrt((((rec - clipped) / rng)[:, :6] ** 2).mean()) < 0.03


def test_conditional_entropy_limits():
    const = np.zeros((10, 50), dtype=np.int64)
    assert conditional_entropy_bits(const) == (0.0, 0.0)
    rng = np.random.default_rng(0)
    iid = rng.integers(0, 4, size=(200, 200))
    h, hc = conditional_entropy_bits(iid)
    assert h == pytest.approx(2.0, abs=0.01) and hc == pytest.approx(2.0, abs=0.02)


def test_smooth_actions_same_motion_at_every_rate():
    lo, hi = smooth_actions(3, 1.0, 10, seed=5), smooth_actions(3, 1.0, 50, seed=5)
    assert np.allclose(lo[:, :, :6], hi[:, ::5, :6])
