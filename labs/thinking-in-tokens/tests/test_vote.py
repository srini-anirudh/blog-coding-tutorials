import pytest

from thinktok.vote import answer_marginal, maj_at_k, pass_at_k, plurality, simulate_vote


def test_plurality():
    assert plurality(["3", "5", "5", "3", "5"]) == "5"
    assert plurality(["7", "8"]) == "7"  # tie: first to reach the top count
    assert plurality([None, None, "1"]) == "1"
    assert plurality([None]) is None


def test_pass_at_k():
    assert pass_at_k(16, 0, 4) == 0.0
    assert pass_at_k(16, 16, 1) == 1.0
    assert pass_at_k(16, 4, 1) == pytest.approx(0.25)
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)


def test_maj_at_full_k_is_the_vote():
    ans = ["1", "2", "2", "3"]
    assert maj_at_k(ans, "2", 4) == 1.0
    assert maj_at_k(ans, "1", 4) == 0.0


def test_vote_recovers_a_mode_below_one_half():
    spread = {"right": 0.4, **{f"w{i}": 0.03 for i in range(20)}}
    concentrated = {"right": 0.4, "w0": 0.45, **{f"w{i}": 0.03 for i in range(1, 6)}}
    assert simulate_vote(spread, "right", 41, trials=2000) > 0.95
    assert simulate_vote(concentrated, "right", 41, trials=2000) < 0.5


def test_answer_marginal():
    assert answer_marginal(["a", "b", "a", "a"]) == [("a", 0.75), ("b", 0.25)]
