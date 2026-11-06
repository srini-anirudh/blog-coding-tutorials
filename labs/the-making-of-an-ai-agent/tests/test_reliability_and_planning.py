from agentcraft.planning import run
from agentcraft.reliability import estimate, exact_pass_hat_k, pass_at_k, pass_hat_k


def test_estimators():
    assert pass_hat_k(10, 10, 4) == 1.0 and pass_hat_k(10, 3, 4) == 0.0
    assert pass_at_k(10, 0, 3) == 0.0 and pass_at_k(10, 8, 3) == 1.0
    assert abs(pass_hat_k(10, 5, 1) - 0.5) < 1e-12


def test_same_mean_different_reliability():
    assert abs(exact_pass_hat_k([0.65] * 10, 8) - 0.65 ** 8) < 1e-12
    assert exact_pass_hat_k([1.0] * 65 + [0.0] * 35, 8) == 0.65
    _, est = estimate([0.5] * 200, n=16, k=2)
    assert abs(est - 0.25) < 0.03


def test_plan_half_life_and_replanning():
    q, L = 0.1, 6
    p = run("plan-then-act", L, q, 20000)
    assert abs(p["success"] - (1 - q) ** L) < 0.015
    r, h = run("react", L, q, 2000), run("hybrid", L, q, 2000)
    assert r["success"] == h["success"] == 1.0
    assert h["calls"] < r["calls"]
