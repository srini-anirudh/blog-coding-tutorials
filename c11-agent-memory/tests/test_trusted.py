"""Our implementations against ones we did not write."""

from agentmem.bench.trusted import compare_index, compare_packer


def test_index_matches_torch_cosine_similarity_and_topk():
    r = compare_index(n=500, queries=20)
    assert r["same_ranking"] == r["queries"]
    assert r["max_score_difference"] < 1e-5


def test_greedy_packer_stays_under_budget_and_near_the_exact_optimum():
    r = compare_packer(trials=40, n=10)
    assert r["worst_ratio"] <= r["mean_ratio"] <= 1.0
    assert r["mean_ratio"] > 0.9   # a measured property of greedy by density, pinned so a regression shows
