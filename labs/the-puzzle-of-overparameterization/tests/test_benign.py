import numpy as np

from overparam.benign import decompose, noise_risk, spiked_model


def test_decomposition_adds_up_and_interpolates():
    model = spiked_model(400)
    rng = np.random.default_rng(0)
    X, _ = model.sample(30, rng)
    eps = model.sigma * rng.standard_normal(30)
    d = decompose(model, X, eps)
    assert d["train_residual"] < 1e-8
    assert d["noise_norm2_tail_share"] > 0.9   # the noise lands mostly in the tail


def test_noise_risk_matches_monte_carlo():
    rng = np.random.default_rng(0)
    spec = np.r_[np.ones(3), np.full(200, 0.05)]
    X = rng.standard_normal((20, spec.size)) * np.sqrt(spec)
    G = X @ X.T
    mc = np.mean([np.sum(spec * (X.T @ np.linalg.solve(G, rng.standard_normal(20))) ** 2) for _ in range(4000)])
    assert abs(noise_risk(X, spec, 1.0) - mc) / mc < 0.05


def test_wide_flat_tail_is_benign_isotropic_is_not():
    rng = np.random.default_rng(0)
    n = 40
    iso = np.ones(2 * n)
    flat = np.r_[1.0, np.full(n * n, 1.0 / n)]
    r_iso = noise_risk(rng.standard_normal((n, iso.size)), iso, 1.0)
    r_flat = noise_risk(rng.standard_normal((n, flat.size)) * np.sqrt(flat), flat, 1.0)
    assert r_iso > 0.7 and r_flat < 0.1
