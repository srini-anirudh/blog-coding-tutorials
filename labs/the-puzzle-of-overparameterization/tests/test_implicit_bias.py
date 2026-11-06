import numpy as np

from overparam.implicit_bias import angle, gd_logistic, hard_margin_svm, margin, separable_data, steepest_descent


def test_svm_matches_brute_force_in_2d():
    X, y, _ = separable_data(40, seed=17)
    w = hard_margin_svm(X, y)
    assert np.isclose(np.min(y * (X @ w)), 1.0)
    th = np.linspace(0, 2 * np.pi, 200001)
    U = np.stack([np.cos(th), np.sin(th)], 1)
    brute = (U @ (X * y[:, None]).T).min(1).max()
    assert abs(margin(w, X, y) - brute) < 1e-6


def test_gd_direction_approaches_svm():
    X, y, _ = separable_data(40, seed=17)
    w_svm = hard_margin_svm(X, y)
    path = gd_logistic(X, y, [100, 10000])
    assert angle(path[10000], w_svm) < angle(path[100], w_svm)
    assert np.all(np.sign(X @ path[10000]) == y)


def test_each_geometry_wins_its_own_margin():
    X, y, _ = separable_data(30, d=50, seed=2, gap=0.3, w_true=np.r_[np.ones(3), np.zeros(47)])
    w = {g: steepest_descent(X, y, g, steps=4000, lr=0.05) for g in ("l2", "l1", "linf")}
    for g, norm in (("l2", 2), ("l1", 1), ("linf", np.inf)):
        others = [margin(w[o], X, y, norm) for o in w if o != g]
        assert margin(w[g], X, y, norm) > max(others)
