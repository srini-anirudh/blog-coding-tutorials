import pytest
import torch

from agentmem.embed import HashingEmbedder, TransformerEmbedder
from agentmem.index_fast import TensorIndex
from agentmem.index_naive import NaiveIndex


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(256)
    a = e.encode(["The deploy region for atlas is eu-west-1.", ""])
    b = e.encode(["The deploy region for atlas is eu-west-1.", ""])
    assert a.shape == (2, 256)
    assert torch.equal(a, b)
    assert torch.allclose(a[0].norm(), torch.tensor(1.0))
    assert a[1].abs().sum() == 0  # no features -> zero vector, never NaN


def test_hashing_similarity_follows_shared_words():
    e = HashingEmbedder()
    q, near, far = e.encode(["which region does atlas deploy to", "atlas deploy region: eu-west-1",
                             "lunch moved to thursday"])
    assert float(q @ near) > 0.3 > float(q @ far)


def _random_unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_tensor_index_matches_naive_index():
    vecs, queries = _random_unit(500, 64), _random_unit(5, 64, seed=1)
    ids = [f"m{i}" for i in range(500)]
    naive, fast = NaiveIndex(64), TensorIndex(64, capacity=16)  # forces several doublings
    naive.add(ids, vecs)
    fast.add(ids, vecs)
    for q in queries:
        a, b = naive.search(q, 10), fast.search(q, 10)
        assert [i for i, _ in a] == [i for i, _ in b]
        assert torch.allclose(torch.tensor([s for _, s in a]), torch.tensor([s for _, s in b]), atol=1e-5)
    batch = fast.search_batch(queries, 10)
    single = fast.search(queries[2], 10)
    assert [i for i, _ in batch[2]] == [i for i, _ in single]  # BLAS may differ in the last float bit


def test_tensor_index_remove_and_reuse_slot():
    vecs = _random_unit(4, 8)
    idx = TensorIndex(8, capacity=4)
    idx.add(["a", "b", "c", "d"], vecs)
    idx.remove("b")
    assert "b" not in [i for i, _ in idx.search(vecs[1], 4)]
    assert len(idx) == 3
    idx.add(["e"], vecs[1:2])                     # reuses b's slot: no growth
    assert idx.matrix.shape[0] == 4
    assert idx.search(vecs[1], 1)[0][0] == "e"


@pytest.mark.network
def test_minilm_mean_pooling_ignores_padding():
    e = TransformerEmbedder()
    short = "atlas deploys to eu-west-1"
    alone = e.encode([short])
    padded = e.encode([short, "a much longer sentence that forces the short one to be padded " * 3])
    assert alone.shape == (1, 384)
    assert torch.allclose(alone[0], padded[0], atol=1e-5)  # padding must not leak into the mean
    q, para, other = e.encode(["which region does atlas deploy to", "atlas ships to the eu-west-1 region",
                               "the design review moved to thursday"])
    assert float(q @ para) > float(q @ other) + 0.2


@pytest.mark.network
def test_minilm_mean_pooling_ignores_padding():
    e = TransformerEmbedder()
    short = "atlas deploy region"
    alone = e.encode([short])
    batched = e.encode([short, "a much longer sentence so that the short one above gets padded in this batch"])
    assert alone.shape == (1, 384)
    assert torch.allclose(alone[0], batched[0], atol=1e-4)
    q, good, bad = e.encode(["where does atlas run in production", "atlas deploy region: us-east-2", "lunch moved to thursday"])
    assert float(q @ good) > float(q @ bad)
