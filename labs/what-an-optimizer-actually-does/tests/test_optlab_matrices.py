import math

import torch

from optlab import BlockQuantized, RMSProp, newton_schulz, ns_scalar, shampoo_direction, spectral_steepest
from optlab.landscapes import Bowl, grad


def test_matrix_iteration_is_scalar_map_on_singular_values():
    g = torch.randn(20, 33, dtype=torch.float64)
    s = torch.linalg.svdvals(g)
    got = torch.linalg.svdvals(newton_schulz(g, dtype=torch.float64))
    want = ns_scalar(s / g.norm()).sort(descending=True).values
    torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


def test_newton_schulz_flattens_the_spectrum_in_bf16():
    sv = torch.linalg.svdvals(newton_schulz(torch.randn(128, 352)).float())
    assert sv.min() > 0.6 and sv.max() < 1.3


def test_spectral_steepest_reaches_the_nuclear_norm_bound():
    g = torch.randn(10, 7, dtype=torch.float64)
    d = spectral_steepest(g)
    assert abs(torch.linalg.matrix_norm(d, ord=2).item() - 1) < 1e-9
    assert abs((g * d).sum().item() + torch.linalg.svdvals(g).sum().item()) < 1e-9
    for _ in range(50):
        r = torch.randn(10, 7, dtype=torch.float64)
        r = r / torch.linalg.matrix_norm(r, ord=2)
        assert (g * r).sum() >= (g * d).sum() - 1e-9


def test_shampoo_is_kronecker_inverse_fourth_root():
    hist = [torch.randn(4, 3, dtype=torch.float64) for _ in range(10)]
    L, R, g = sum(h @ h.T for h in hist), sum(h.T @ h for h in hist), hist[-1]
    w, q = torch.linalg.eigh(torch.kron(L, R))
    want = q @ torch.diag(w**-0.25) @ q.T @ g.reshape(-1)
    torch.testing.assert_close(shampoo_direction(L, R, g).reshape(-1), want)


def test_block_quantization_limits_outlier_damage():
    x = torch.randn(2**16)
    assert ((BlockQuantized(x, 2048, "log").dequantize() - x).norm() / x.norm()) < 0.05
    x[0] = 1e6  # beyond the log code's 16 octaves for every other entry
    lost = {b: (BlockQuantized(x, b, "log").dequantize() == 0).float().mean().item() for b in (None, 2048)}
    assert lost[None] > 0.99  # one scale for the tensor: everything else rounds to 0
    assert lost[2048] < 2048 / 2**16 + 1e-3  # block-wise: about one block (the outlier's own) is lost


def test_diagonal_preconditioner_prefers_axis_aligned_curvature():
    def steps_to_tol(land):
        p = torch.nn.Parameter(torch.tensor(land.start))
        opt = RMSProp([p], 0.5623)
        for i in range(300):
            g, f = grad(land, p)
            if f < 1e-3:
                return i
            p.grad = g
            opt.step()
        return 300

    assert steps_to_tol(Bowl("a")) < 20 < steps_to_tol(Bowl("r", angle=math.pi / 4))
