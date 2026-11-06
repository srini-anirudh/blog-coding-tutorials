import torch

from agentmem.embed import HashingEmbedder
from agentmem.records import MemoryItem
from agentmem.retrieval import HybridRetriever, HybridWeights


def _retriever(alpha, beta, gamma):
    r = HybridRetriever(HashingEmbedder(), HybridWeights(alpha, beta, gamma, half_life=2.0))
    r.add(MemoryItem("semantic", "atlas deploy region: eu-west-1", t=1.0, importance=0.2, id="old-relevant"))
    r.add(MemoryItem("semantic", "lunch moved to thursday", t=9.0, importance=0.1, id="new-irrelevant"))
    r.add(MemoryItem("episodic", "never deploy on fridays", t=5.0, importance=1.0, id="important"))
    return r


def test_each_weight_alone_ranks_by_its_own_term():
    q = "which region does atlas deploy to"
    assert _retriever(1, 0, 0).search(q, now=10.0, k=1)[0].item.id == "old-relevant"
    assert _retriever(0, 1, 0).search(q, now=10.0, k=1)[0].item.id == "new-irrelevant"
    assert _retriever(0, 0, 1).search(q, now=10.0, k=1)[0].item.id == "important"


def test_score_is_the_weighted_sum_of_its_terms():
    r = _retriever(1.0, 0.5, 0.3)
    for hit in r.search("atlas region", now=10.0, k=3):
        assert abs(hit.score - (1.0 * hit.semantic + 0.5 * hit.recency + 0.3 * hit.importance)) < 1e-5
        age = 10.0 - hit.item.t
        assert abs(hit.recency - 0.5 ** (age / 2.0)) < 1e-6
        assert 0.0 <= hit.semantic <= 1.0


def test_removed_items_never_come_back():
    r = _retriever(1, 0, 0)
    r.remove("old-relevant")
    assert all(h.item.id != "old-relevant" for h in r.search("atlas region", 10.0, k=5))


def test_mmr_skips_near_duplicates():
    r = HybridRetriever(HashingEmbedder(), HybridWeights(1, 0, 0))
    for i in range(3):
        r.add(MemoryItem("semantic", "atlas deploy region: eu-west-1", t=1.0, id=f"dup{i}"))
    r.add(MemoryItem("semantic", "atlas owner: priya", t=1.0, id="owner"))
    plain = [h.item.id for h in r.search("atlas deploy region owner", 2.0, k=2)]
    diverse = [h.item.id for h in r.search("atlas deploy region owner", 2.0, k=2, mmr_lambda=0.5)]
    assert plain[1].startswith("dup") and "owner" in diverse
