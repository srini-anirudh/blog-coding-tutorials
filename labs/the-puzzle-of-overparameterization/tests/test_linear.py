import numpy as np

from overparam.linear import bias_variance, epoch_wise, harmonic_model, model_wise, spectral_fit, tuned_ridge


def test_spectral_fit_matches_pinv_ridge_and_gd():
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((20, 50)), rng.standard_normal(20)
    assert np.allclose(spectral_fit(X, y), np.linalg.pinv(X) @ y)
    lam = 0.3
    assert np.allclose(spectral_fit(X, y, ridge=lam), np.linalg.solve(X.T @ X + lam * np.eye(50), X.T @ y))
    beta, lr = np.zeros(50), 1.0 / np.linalg.norm(X, 2) ** 2
    for _ in range(25):
        beta -= lr * X.T @ (X @ beta - y)
    assert np.allclose(spectral_fit(X, y, steps=25), beta)


def test_min_norm_interpolates_when_overparameterized():
    rng = np.random.default_rng(1)
    X, y = rng.standard_normal((10, 30)), rng.standard_normal(10)
    assert np.allclose(X @ spectral_fit(X, y), y)


def test_bias_variance_sums_and_trends():
    rows = bias_variance([1, 3, 12], redraws=200)
    assert rows[0]["bias2"] > rows[1]["bias2"]            # degree 1 underfits
    assert rows[2]["variance"] > 10 * rows[1]["variance"]  # degree 12 on 14 points jitters


def test_peak_at_interpolation_threshold():
    rows = {r["p"]: r["risk_median"] for r in model_wise([10, 40, 420], trials=15)}
    assert rows[40] > 5 * rows[10] and rows[40] > 5 * rows[420]


def test_tuned_ridge_removes_peak():
    tuned = {r["p"]: r["risk_mean"] for r in tuned_ridge([10, 40, 420], trials=10, ridges=np.logspace(-3, 1, 9))}
    assert tuned[40] < 2 * tuned[10]


def test_epoch_wise_rises_to_min_norm():
    rows, end = epoch_wise([10, 100_000], trials=10)
    assert rows[0]["risk_median"] < rows[1]["risk_median"]
    assert abs(rows[1]["risk_median"] - end) / end < 0.05


def test_harmonic_model_unit_signal():
    m = harmonic_model()
    assert np.isclose(np.sum(m.spec * m.beta_star**2), 1.0)
    assert m.risk(m.beta_star) == 0.0
