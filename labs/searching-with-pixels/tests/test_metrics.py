import math

import pytest

from pixsearch import metrics as M


def test_reciprocal_rank_worked_example():
    # The post: a correct result at rank 1 gives 1, at rank 4 it gives 0.25.
    assert M.reciprocal_rank([7, 1, 2], {7}) == 1.0
    assert M.reciprocal_rank([3, 1, 2, 9, 5], {9}) == 0.25
    assert M.reciprocal_rank([3, 1], {9}) == 0.0


def test_recall_and_precision():
    assert M.recall_at_k([5, 4, 3], {3}, 2) == 0.0
    assert M.recall_at_k([5, 4, 3], {3}, 3) == 1.0
    assert M.precision_at_1([5, 4], {5, 4}) == 1.0


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
def test_ndcg_one_relevant_page(rank):
    # The post: with one relevant page, nDCG@5 = 1 / log2(1 + rank) inside the top five.
    ranking = [10, 11, 12, 13, 14]
    ranking[rank - 1] = 99
    assert M.ndcg_at_k(ranking, {99}, 5) == pytest.approx(1 / math.log2(1 + rank))


def test_ndcg_outside_cutoff_and_rank3_half():
    assert M.ndcg_at_k([1, 2, 3, 4, 5, 99], {99}, 5) == 0.0
    # Rank 3 counts half, because log2(4) = 2.
    assert M.ndcg_at_k([1, 2, 99], {99}, 5) == pytest.approx(0.5)


def test_ndcg_graded_by_hand():
    rel = {"a": 3, "b": 2, "c": 1}
    ranking = ["c", "a", "x", "b"]
    dcg = (2**1 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3) + 0 + (2**2 - 1) / math.log2(5)
    idcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
    assert M.ndcg_at_k(ranking, rel, 5) == pytest.approx(dcg / idcg)
    assert M.ndcg_at_k(["a", "b", "c"], rel, 5) == pytest.approx(1.0)


def test_evaluate_averages_in_percent():
    out = M.evaluate([[1, 2], [2, 1]], [{1}, {1}], ks=(1, 2))
    assert out["R@1"] == 50.0 and out["R@2"] == 100.0
    assert out["MRR"] == pytest.approx(75.0)
