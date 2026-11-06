"""Scaling in space: Adagrad, RMSProp and Adam, and the details that change what they do.

Claims checked:
  1. Without bias correction, the first Adam step is about 3.16x too large (b1=0.9, b2=0.999).
  2. Adam's update has RMS near 1 per coordinate, whatever the gradient scale.
  3. Adagrad's step decays like 1/sqrt(t); RMSProp's does not.
  4. m/(sqrt(v)+eps) and m/sqrt(v+eps) differ for small gradients; the first falls back to SGD with lr/eps.
  5. Adam + L2 regularizes coordinates with large gradients less; AdamW decays every coordinate the same.

Run: python labs/what-an-optimizer-actually-does/scripts/03_adaptive.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optlab import Adagrad, Adam, RMSProp, rms
from optlab.common import Report


def steps_of(make, grads):
    """Feed a sequence of gradients to one parameter; return |change| / lr per step."""
    p = torch.nn.Parameter(torch.zeros_like(grads[0]))
    opt = make([p])
    out = []
    for g in grads:
        before = p.detach().clone()
        p.grad = g
        opt.step()
        out.append(((p.detach() - before).abs().mean() / opt.lr).item())
    return out


def main():
    rep = Report("03_adaptive")
    torch.manual_seed(0)

    rep("## 1. Bias correction, constant gradient g = 1, b1 = 0.9, b2 = 0.999: |step| / lr")
    const = [torch.ones(1)] * 3000
    raw = steps_of(lambda p: Adam(p, lr=1e-3, bias_correction=False), const)
    cor = steps_of(lambda p: Adam(p, lr=1e-3), const)
    for t in (1, 2, 5, 10, 20, 50, 100, 300, 1000, 3000):
        rep(f"t={t:<5} uncorrected {raw[t - 1]:6.3f}   corrected {cor[t - 1]:6.3f}")
    peak = max(range(len(raw)), key=raw.__getitem__)
    below = next(t for t in range(peak, len(raw)) if raw[t] < 1.5)
    rep(f"uncorrected peak {raw[peak]:.2f} at t={peak + 1}; it stays above 1.5x until t={below + 1}")

    rep("\n## 2. RMS of the Adam direction m_hat/(sqrt(v_hat)+eps) after 2000 steps, 10k coordinates")
    rep("gradient per coordinate = scale * (mean + noise), noise ~ N(0, 1), mean ~ +-snr")
    rep(f"{'scale':>8} " + " ".join(f"{'snr ' + str(s):>9}" for s in (0, 0.1, 0.3, 1, 3, 10)))
    rms_table = {}
    for scale in (1e-4, 1.0, 1e4):
        row = []
        for snr in (0, 0.1, 0.3, 1, 3, 10):
            gen = torch.Generator().manual_seed(1)
            mean = snr * torch.sign(torch.randn(10_000, generator=gen))
            opt = Adam([torch.nn.Parameter(torch.zeros(10_000))], lr=1e-3)
            p = opt.params[0]
            for _ in range(2000):
                u = opt.direction(p, scale * (mean + torch.randn(10_000, generator=gen)), opt.state.setdefault(p, {}))
            row.append(rms(u).item())
        rms_table[scale] = row
        rep(f"{scale:>8.0e} " + " ".join(f"{r:>9.3f}" for r in row))
    rep(f"pure-noise prediction sqrt((1-b1)/(1+b1)) = {((1 - 0.9) / (1 + 0.9)) ** 0.5:.3f}")

    rep("\n## 3. Step size over time, random-sign gradients of magnitude 1: |step| / lr")
    signs = [torch.sign(torch.randn(1000)) for _ in range(10_000)]
    ada = steps_of(lambda p: Adagrad(p, lr=1e-2), signs)
    rmsp = steps_of(lambda p: RMSProp(p, lr=1e-2, beta2=0.99), signs)
    for t in (1, 10, 100, 1000, 10_000):
        rep(f"t={t:<6} Adagrad {ada[t - 1]:.4f} (1/sqrt(t) = {t ** -0.5:.4f})   RMSProp {rmsp[t - 1]:.4f}")

    rep("\n## 4. Where eps goes: steady-state |update| for a constant gradient g, eps = 1e-8")
    rep(f"{'g':>8} {'m/(sqrt v + eps)':>17} {'m/sqrt(v + eps)':>16} {'g/eps (SGD, lr/eps)':>20}")
    for g in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        out = []
        for inside in (False, True):
            opt = Adam([torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))], lr=1.0, eps_inside=inside)
            p = opt.params[0]
            for _ in range(5000):
                u = opt.direction(p, torch.full((1,), g, dtype=torch.float64), opt.state.setdefault(p, {}))
            out.append(u.item())
        rep(f"{g:>8.0e} {out[0]:>17.3e} {out[1]:>16.3e} {g / 1e-8:>20.3e}")

    rep("\n## 5. Coupled L2 vs decoupled decay: zero-mean noisy gradients, lr 1e-3, wd 0.1, 1000 steps")
    rep("1000 coordinates of type A (gradient noise std 10) and 1000 of type B (std 0.01), all start at 1.0")
    std = torch.cat([torch.full((1000,), 10.0), torch.full((1000,), 0.01)])
    for decoupled in (True, False):
        gen = torch.Generator().manual_seed(2)
        p = torch.nn.Parameter(torch.ones(2000))
        opt = Adam([p], lr=1e-3, weight_decay=0.1, decoupled=decoupled)
        for _ in range(1000):
            p.grad = std * torch.randn(2000, generator=gen)
            opt.step()
        name = "AdamW" if decoupled else "Adam + L2"
        rep(f"{name:<10} mean A -> {p[:1000].mean().item():.4f}   mean B -> {p[1000:].mean().item():.4f}"
            f"   (pure decay exp(-0.1) = 0.9048)")
    rep.save({"bias_correction_peak": [peak + 1, raw[peak]], "update_rms": {str(k): v for k, v in rms_table.items()}})


if __name__ == "__main__":
    main()
