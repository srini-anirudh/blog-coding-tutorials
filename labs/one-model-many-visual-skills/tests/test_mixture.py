import pytest

from skillmix.mixture import Source, cap, pareto_front, pick_constrained, pick_weighted, shares, signal_share


def test_post_normalization_example():
    w, n = {"vqa": 0.5, "caption": 0.5}, {"vqa": 2, "caption": 150}
    assert signal_share(w, n, "token")["vqa"] == pytest.approx(1 / 76)  # 0.5*2 / (0.5*2 + 0.5*150) = 1.3%
    assert signal_share(w, n, "example")["vqa"] == pytest.approx(0.5)


def test_cap_and_shares():
    srcs = [Source("a", 1000, 10, 0, 1), Source("b", 100, 100, 0, 10)]
    capped = cap(srcs, 200)
    assert [s.examples for s in capped] == [200, 100]
    sh = shares(srcs)
    assert sh["a"]["examples"] == pytest.approx(1000 / 1100)
    assert sh["a"]["compute"] == pytest.approx(0.5) and sh["a"]["loss_tokens"] == pytest.approx(0.5)


def test_pareto_and_picks():
    pts = [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.4, 0.4), (0.5, 0.2)]
    assert pareto_front(pts) == [0, 1, 2]
    assert pick_weighted(pts, (1, 0)) == 0 and pick_weighted(pts, (0, 1)) == 2
    assert pick_constrained(pts, maximise=1, floors={0: 0.45}) == 1
    assert pick_constrained(pts, maximise=1, floors={0: 0.95}) is None
