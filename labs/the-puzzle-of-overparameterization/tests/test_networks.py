import torch

from overparam import bits, grokking, lazy_rich, memorize, symmetry


def test_permutation_and_rescaling_keep_the_function():
    x, y = symmetry.regression_data()
    model = symmetry.small_mlp(16)
    symmetry.train(model, x, y, steps=200)
    grid = torch.linspace(-1, 1, 200)[:, None]
    with torch.no_grad():
        ref = model(grid)
        assert torch.allclose(symmetry.permuted(model, torch.randperm(16))(grid), ref, atol=1e-5)
        assert torch.allclose(symmetry.rescaled(model, 5.0)(grid), ref, atol=1e-4)
    h1 = symmetry.top_hessian_eigenvalue(model, x, y, iters=30)
    h5 = symmetry.top_hessian_eigenvalue(symmetry.rescaled(model, 5.0), x, y, iters=30)
    assert h5 > 3 * h1


def test_tangent_features_match_autograd():
    model = lazy_rich.TwoLayer(5, 8)
    x = torch.randn(3, 5)
    phi = lazy_rich.tangent_features(model, x)
    for i in range(3):
        g = torch.autograd.grad(model(x[i:i + 1]).sum(), [model.a, model.w])
        assert torch.allclose(phi[i], torch.cat([g[0], g[1].flatten()]), atol=1e-6)


def test_large_alpha_is_lazy():
    lazy = lazy_rich.train(100.0, m=256, n=50, steps=300)
    rich = lazy_rich.train(0.1, m=256, n=50, steps=300)
    assert lazy["rel_weight_change"] < 0.01 < rich["rel_weight_change"]


def test_mlp_memorizes_random_labels():
    x, y, xt, yt = memorize.synthetic_clusters(300, 300)
    real = memorize.fit(x, y, xt, yt, width=128, max_epochs=200)
    rand = memorize.fit(x, memorize.corrupt_labels(y, 1.0), xt, yt, width=128, max_epochs=300)
    assert rand["train_acc"] == 1.0 and rand["test_acc"] < 0.25
    assert real["test_acc"] > 0.8


def test_corrupt_labels_fraction():
    y = torch.zeros(1000, dtype=torch.long)
    changed = (memorize.corrupt_labels(y, 0.5) != y).float().mean()
    assert 0.4 < changed < 0.5   # half are redrawn, a tenth of those land on the old label


def test_grokking_pieces():
    xtr, ytr, xva, yva = grokking.mod_add_split(13, 0.5)
    assert len(xtr) + len(xva) == 169 and torch.all((xtr.sum(1) % 13) == ytr)
    w = torch.zeros(13, 4)
    w[:, 0] = torch.cos(2 * torch.pi * 3 * torch.arange(13) / 13)
    assert grokking.fourier_concentration(w, top=1) > 0.99
    rows, _ = grokking.train(p=13, steps=50, log_every=25)
    assert rows[-1]["train_acc"] > rows[0]["train_acc"]


def test_bounds_arithmetic():
    assert 300 < bits.vc_bound()["bound_term"] < 340
    assert 780 < bits.capacity_budget()["corpus_over_capacity"] < 800
    assert bits.occam_bound(0.0, 0, 10**6) < 0.01


def test_compression_shrinks_description():
    x, y = bits.xor_task(2000)
    model = bits.train_classifier(x, y, epochs=3)
    _, b8 = bits.compress(model, 8)
    _, b3 = bits.compress(model, 3, keep=0.2)
    assert b3 < b8 < 32 * sum(p.numel() for p in model.parameters())
