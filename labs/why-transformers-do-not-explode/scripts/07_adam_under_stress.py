"""Part VII: the optimizer is part of the dynamical system. Adam's first step, stale second moments, eps,
clipping, scale invariance under normalization, and gradient noise against batch size.

    python labs/why-transformers-do-not-explode/scripts/07_adam_under_stress.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from stablelab.common import Report, seed_all  # noqa: E402
from stablelab.optim import InspectableAdam, clip_global_norm, stress_stream  # noqa: E402

WINDOWS = {"step 1": (0, 1), "hot start 2-20": (1, 20), "ordinary 300-399": (299, 399),
           "quiet 1000-1499": (999, 1499), "burst 1500-1520": (1499, 1520), "burst 1540-1559": (1539, 1559)}


def run_stream(b2: float, eps: float, warmup: int, lr: float = 1e-3) -> dict:
    grads = stress_stream()
    opt = InspectableAdam(grads.shape[1], lr=lr, betas=(0.9, b2), eps=eps)
    ratio = []
    for t, g in enumerate(grads, start=1):
        step_lr = lr * min(1.0, t / warmup) if warmup else lr
        out = opt.step(g, lr=step_lr)
        ratio.append((out["update"].abs() / lr).mean().item())  # |update| in units of the base lr
    return {name: max(ratio[a:b]) if name.startswith("burst") else sum(ratio[a:b]) / (b - a)
            for name, (a, b) in WINDOWS.items()}


def main() -> None:
    seed_all(0)
    r = Report("07_adam_under_stress")

    r("1. InspectableAdam against torch.optim.Adam on 200 random gradients:")
    p = torch.zeros(256, dtype=torch.float64, requires_grad=True)
    torch_opt = torch.optim.Adam([p], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    ours = InspectableAdam(256, lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    mine = torch.zeros(256, dtype=torch.float64)
    for _ in range(200):
        g = torch.randn(256, dtype=torch.float64)
        p.grad = g.clone()
        torch_opt.step()
        mine += ours.step(g)["update"]
    r(f"   max |difference in parameters| = {(p.detach() - mine).abs().max().item():.2e}")
    for size in (1.0, 1e-3, 1e-6):
        first = InspectableAdam(256, lr=1e-3).step(torch.randn(256, dtype=torch.float64) * size)["update"].abs() / 1e-3
        r(f"   Step 1, gradients of size {size:g}, eps 1e-8: |update| / lr in [{first.min().item():.4f},"
          f" {first.max().item():.4f}]")
    r("   Adam's first step is lr * sign(g) whatever the gradient size, until |g| gets within a few orders of eps.")
    r()

    r("2. The synthetic stream of the post's Figure 10 (256 coordinates): hot start (std 10, steps 1-20), ordinary")
    r("   (std 1), quiet stretch from step 400 (std 0.01), correlated burst at steps 1500-1559 (mean 1).")
    r("   Each cell is |update| / lr averaged over coordinates: the mean over the window, or the peak for burst windows.")
    rows = []
    for b2, eps, warmup in ((0.999, 1e-8, 0), (0.95, 1e-8, 0), (0.999, 1e-8, 300), (0.999, 1e-2, 0), (0.95, 1e-2, 0)):
        res = run_stream(b2, eps, warmup)
        rows.append([b2, eps, warmup] + list(res.values()))
    r.table(["beta2", "eps", "warmup", *WINDOWS], rows)
    r("beta2 = 0.999: the quiet stretch steps about 100x under the beta2 = 0.95 run, because sqrt(v_hat) still")
    r("remembers the loud start. The burst overshoots lr only modestly here (peak 1.18x), because that same memory")
    r("of the hot start keeps the denominator large. Section 3 removes the hot start and shows the clean overshoot.")
    r("Warmup only changes the first 300 steps. eps = 1e-2 does nothing at beta2 = 0.999 (the remembered denominator")
    r("is far above eps) and halves the quiet-stretch steps at beta2 = 0.95, where sqrt(v_hat) is about 0.01.")
    r()

    r("3. The rare-token row: gradient exactly 0 for 5,000 steps, then gradient 1 every step. Peak |update| / lr.")
    rows = []
    for b2 in (0.999, 0.99, 0.95):
        opt = InspectableAdam(1, lr=1.0, betas=(0.9, b2))
        for _ in range(5000):
            opt.step(torch.zeros(1, dtype=torch.float64))
        ups = [opt.step(torch.ones(1, dtype=torch.float64))["update"].abs().item() for _ in range(200)]
        peak = max(range(200), key=lambda i: ups[i])
        rows.append([b2, ups[0], max(ups), peak + 1])
    r.table(["beta2", "first nonzero step", "peak", "steps after the gradient appears"], rows)
    r("The first step is (1 - beta1) / sqrt(1 - beta2) times lr: 3.2x at beta2 = 0.999. The peak, 12 steps later, is"
      " 6.5x: m_hat fills in ~10 steps while v_hat has only seen 12 of its ~1,000-step window.")
    r()

    r("4. Global-norm clipping keeps the direction; per-coordinate clamping does not. Two tensors, tau = 1.")
    g1, g2 = torch.randn(1000, dtype=torch.float64) * 5, torch.randn(10, dtype=torch.float64) * 50
    (c1, c2), total = clip_global_norm([g1, g2], 1.0)
    full, clipped = torch.cat([g1, g2]), torch.cat([c1, c2])
    clamped = full.clamp(-1, 1)
    r(f"   ||g|| = {total:.1f} -> {clipped.norm().item():.4f}; cosine(g, clipped) = {F.cosine_similarity(full, clipped, 0).item():.6f};"
      f" cosine(g, clamp(g, -1, 1)) = {F.cosine_similarity(full, clamped, 0).item():.4f}")
    r()

    r("5. Scale invariance: W feeds a LayerNorm, so L(cW) = L(W) but ||dL/dW|| scales like 1/c.")
    x, target = torch.randn(64, 32), torch.randn(64, 32)
    w0 = torch.randn(32, 32) / 32**0.5
    rows = []
    for c in (0.5, 1.0, 2.0, 4.0):
        w = (c * w0).requires_grad_(True)
        loss = F.mse_loss(F.layer_norm(x @ w.T, (32,)), target)
        loss.backward()
        rows.append([c, loss.item(), w.grad.norm().item(), w.grad.norm().item() * c])
    r.table(["c", "loss", "||grad||", "c * ||grad||"], rows)
    r("The effective learning rate on W's direction is lr / ||W||^2, so weight decay, which shrinks ||W||, raises it.")
    r()

    r("6. Gradient noise against batch size, linear regression with 4,096 examples: Var(g_B) ~ 1 / B.")
    xs = torch.randn(4096, 16, dtype=torch.float64)
    ys = xs @ torch.randn(16, dtype=torch.float64) + torch.randn(4096, dtype=torch.float64)
    w = torch.zeros(16, dtype=torch.float64)
    per_example = -2 * (ys - xs @ w)[:, None] * xs
    rows = []
    for B in (8, 32, 128, 512):
        means = torch.stack([per_example[torch.randint(4096, (B,))].mean(0) for _ in range(2000)])
        var = means.var(0).sum().item()
        rows.append([B, var, var * B])
    r.table(["B", "total variance of minibatch gradient", "variance * B"], rows)
    r.save()


if __name__ == "__main__":
    main()
