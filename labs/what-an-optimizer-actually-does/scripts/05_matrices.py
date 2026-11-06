"""Matrices, not vectors: steepest descent under the spectral norm, Newton-Schulz, and Shampoo's exponent.

Claims checked:
  1. argmin <G, D> over sigma_max(D) <= 1 is -U V^T: no other direction of spectral norm 1 descends more.
  2. Muon's matrix iteration is the scalar map s <- a s + b s^3 + c s^5 on every singular value.
  3. With Muon's coefficients, 5 steps lift tiny singular values into a band around 1 (the post says
     roughly [0.7, 1.3]). The textbook cubic iteration needs many more steps from the same start.
  4. bf16 is enough: the bf16 result is close to the fp32 one.
  5. RMS(U V^T) = 1/sqrt(max(m, n)), so Muon rescales by 0.2 sqrt(max(m, n)) to get RMS 0.2.
  6. Shampoo's L^-1/4 G R^-1/4, written as one matrix acting on vec(G).

Run: python labs/what-an-optimizer-actually-does/scripts/05_matrices.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optlab import CUBIC_COEFFS, newton_schulz, ns_scalar, rms, shampoo_direction, spectral_steepest
from optlab.common import Report


def gradient_like(m, n, seed=0, decay=1.0):
    """A matrix with singular values i^-decay and random singular vectors: a few strong directions."""
    gen = torch.Generator().manual_seed(seed)
    u, _ = torch.linalg.qr(torch.randn(m, min(m, n), generator=gen, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(n, min(m, n), generator=gen, dtype=torch.float64))
    s = torch.arange(1, min(m, n) + 1, dtype=torch.float64) ** -decay
    return u @ torch.diag(s) @ v.T, s


def main():
    rep = Report("05_matrices")
    G, s = gradient_like(64, 176)

    rep("## 1. Descent <G, D> per unit spectral norm, G is 64 x 176 with singular values 1/i")
    nuclear = s.sum().item()
    cands = {
        "-U V^T (exact)": spectral_steepest(G),
        "-Newton-Schulz(G), 5 steps fp32": -newton_schulz(G.float(), dtype=torch.float32).double(),
        "-G (SGD direction)": -G,
        "-sign(G) (sign descent)": -torch.sign(G),
    }
    for name, D in cands.items():
        D = D / torch.linalg.matrix_norm(D, ord=2)  # scale to spectral norm exactly 1
        rep(f"{name:<34} {(G * D).sum().item():>9.4f}")
    gen = torch.Generator().manual_seed(3)
    best = min(((G * (R / torch.linalg.matrix_norm(R, ord=2))).sum().item()
                for R in (torch.randn(64, 176, generator=gen, dtype=torch.float64) for _ in range(2000))))
    rep(f"{'best of 2000 random directions':<34} {best:>9.4f}")
    rep(f"{'bound: -(nuclear norm of G)':<34} {-nuclear:>9.4f}")

    rep("\n## 2. Matrix iteration vs scalar map on the singular values (fp64)")
    X = newton_schulz(G, dtype=torch.float64)
    s_mat = torch.linalg.svdvals(X)
    s_map = ns_scalar(s / G.norm()).sort(descending=True).values
    rep(f"max |sigma_i(NS(G)) - map(sigma_i / |G|_F)| = {(s_mat - s_map).abs().max().item():.2e}")

    rep("\n## 3. The scalar map from a range of starting singular values")
    starts = torch.tensor([1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.6, 1.0], dtype=torch.float64)
    rep("start    " + " ".join(f"{x:>8.0e}" for x in starts))
    for k in range(1, 6):
        rep(f"Muon k={k}  " + " ".join(f"{x:>8.3f}" for x in ns_scalar(starts, k)))
    rep("cubic k=5 " + " ".join(f"{x:>8.3f}" for x in ns_scalar(starts, 5, CUBIC_COEFFS)))
    for x0 in (1e-3, 1e-2):
        for name, co in (("Muon", None), ("cubic", CUBIC_COEFFS)):
            x, k = torch.tensor(x0, dtype=torch.float64), 0
            while x < 0.7 and k < 200:
                x = ns_scalar(x, 1) if co is None else ns_scalar(x, 1, co)
                k += 1
            rep(f"steps for {x0:.0e} to reach 0.7: {name:<5} {k}")
    dense = torch.logspace(-3, 0, 100_000, dtype=torch.float64)
    out = ns_scalar(dense, 5)
    rep(f"after 5 Muon steps, starts in [1e-3, 1]: output in [{out.min():.3f}, {out.max():.3f}]")
    dense2 = torch.logspace(-2.5, 0, 100_000, dtype=torch.float64)
    out2 = ns_scalar(dense2, 5)
    rep(f"after 5 Muon steps, starts in [3e-3, 1]: output in [{out2.min():.3f}, {out2.max():.3f}]")

    rep("\n## 4. Singular values of the Newton-Schulz output, bf16 vs fp32 (5 steps)")
    for name, (m, n, decay) in {"random Gaussian 256 x 704": (256, 704, 0.0),
                                "spectrum 1/i 256 x 704": (256, 704, 1.0)}.items():
        A = gradient_like(m, n, decay=decay)[0].float() if decay else torch.randn(m, n, generator=gen)
        o16 = newton_schulz(A).float()
        o32 = newton_schulz(A, dtype=torch.float32)
        sv = torch.linalg.svdvals(o16)
        inside = ((sv > 0.7) & (sv < 1.3)).float().mean().item()
        rel = ((o16 - o32).norm() / o32.norm()).item()
        rep(f"{name:<27} bf16 sigma in [{sv.min():.3f}, {sv.max():.3f}], {inside:.0%} in (0.7, 1.3); "
            f"|bf16 - fp32| / |fp32| = {rel:.3f}")

    rep("\n## 5. RMS of the orthogonalized matrix and of Muon's rescaled step")
    for m, n in ((128, 128), (128, 352), (352, 128), (1024, 2816)):
        o = newton_schulz(torch.randn(m, n, generator=gen)).float()
        rep(f"{m:>5} x {n:<5} RMS(O) {rms(o).item():.4f}  1/sqrt(max) {max(m, n) ** -0.5:.4f}  "
            f"RMS(0.2 sqrt(max) O) {rms(0.2 * max(m, n) ** 0.5 * o).item():.4f}")

    rep("\n## 6. Shampoo: vec(L^-1/4 G R^-1/4) as a Kronecker preconditioner, G is 6 x 4")
    g = torch.randn(6, 4, generator=gen, dtype=torch.float64)
    hist = [torch.randn(6, 4, generator=gen, dtype=torch.float64) for _ in range(20)] + [g]
    L = sum(h @ h.T for h in hist)
    R = sum(h.T @ h for h in hist)
    d = shampoo_direction(L, R, g).reshape(-1)  # row-major vec
    K = torch.kron(L, R)
    w, q = torch.linalg.eigh(K)
    for p in (-0.5, -0.25):
        Kp = q @ torch.diag(w**p) @ q.T
        rep(f"|vec(L^-1/4 G R^-1/4) - (L kron R)^{p} vec(G)| / |.| = {((d - Kp @ g.reshape(-1)).norm() / d.norm()).item():.2e}")
    rep("so the quarter powers give (L kron R)^-1/4 = (L^1/2 kron R^1/2)^-1/2: Adagrad's inverse square root")
    rep("of the Kronecker estimate L^1/2 kron R^1/2 (each of L and R already holds one factor of g^2).")
    rep.save()


if __name__ == "__main__":
    main()
