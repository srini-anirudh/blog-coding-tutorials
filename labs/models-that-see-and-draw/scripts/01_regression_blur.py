"""Why you cannot regress pixels: squared error returns the mean image; a sampler commits to one mode.

Data: one caption ("a cat sitting on a sofa"), 8 x 8 images, the cat on the left or on the right, at one of
three heights. Three generators are trained on the same images:

1. regression: f(c) trained with squared error. Its optimum is E[x | c].
2. rectified flow: the same squared error, but on the velocity at a noisy point, then a 100-step sampler.
3. raster autoregression over binary pixels: a softmax per pixel, 64 passes per image.

For each, the script reports how sharp the images are and whether each one commits to one side.
"""

from __future__ import annotations

import time

import _shared  # noqa: F401  (sets the import path)
import torch
import torch.nn.functional as F

from seedraw import data, flow
from seedraw.common import Report, ascii_image, seed_all
from seedraw.priors import TokenPrior, train_prior

N_SAMPLES = 1000
FLOW_STEPS = 8000


def describe(name: str, x: torch.Tensor, train: torch.Tensor, rep: Report) -> dict:
    """Sharpness, commitment to one side, and distance to the nearest valid image."""
    x = x.clamp(0, 1)
    bal = data.side_balance(x)
    committed = ((bal < 0.1) | (bal > 0.9)).float()
    valid = torch.unique(train.flatten(1), dim=0)  # the 6 distinct training images
    nearest = torch.cdist(x.flatten(1), valid).min(dim=1).values.pow(2) / x[0].numel()
    stats = {
        "max_pixel_mean": float(x.flatten(1).max(1).values.mean()),
        "committed_to_one_side": float(committed.mean()),
        "left_share_of_committed": float((bal[committed.bool()] > 0.5).float().mean()) if committed.any() else None,
        "mse_to_nearest_valid_image": float(nearest.mean()),
    }
    rep(f"{name}: brightest pixel {stats['max_pixel_mean']:.2f}, committed to one side "
        f"{stats['committed_to_one_side']:.1%}, left among committed {stats['left_share_of_committed'] or 0:.1%}, "
        f"MSE to nearest valid image {stats['mse_to_nearest_valid_image']:.4f}")
    return stats


def main() -> None:
    seed_all(0)
    rep = Report("01_regression_blur")
    g = torch.Generator().manual_seed(0)
    train = data.cat_images(4000, g)
    c = torch.zeros(len(train), dtype=torch.long)  # one caption for every image

    rep("Valid images: 6 (2 sides x 3 heights). Two of them:")
    rep(ascii_image(train[data.side_balance(train) > 0.5][0]))
    rep(ascii_image(train[data.side_balance(train) < 0.5][0]))
    rep()

    # 1. Regression with squared error.
    t0 = time.time()
    reg = torch.nn.Sequential(torch.nn.Embedding(1, 32), torch.nn.Linear(32, 128), torch.nn.SiLU(), torch.nn.Linear(128, 64))
    opt = torch.optim.Adam(reg.parameters(), lr=3e-3)
    for _ in range(1500):
        i = torch.randint(0, len(train), (256,))
        loss = F.mse_loss(reg(c[i]).view(-1, 8, 8), train[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        f_c = reg(c[:1]).view(1, 8, 8)
    gap = float((f_c[0] - train.mean(0)).abs().max())
    rep(f"[1] Regression, 1500 steps ({time.time() - t0:.1f} s). Output f(c):")
    rep(ascii_image(f_c[0]))
    rep(f"Largest gap between f(c) and the mean training image: {gap:.4f}  (the optimum is E[x | c])")
    results = {"regression": describe("regression", f_c.expand(N_SAMPLES, 8, 8), train, rep)}
    results["regression"]["max_gap_to_mean_image"] = gap
    rep()

    # 2. Rectified flow.
    t0 = time.time()
    v = flow.VelocityMLP(64, hidden=512)
    opt = torch.optim.Adam(v.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FLOW_STEPS)
    for _ in range(FLOW_STEPS):
        i = torch.randint(0, len(train), (256,))
        loss = flow.flow_loss(v, train[i].flatten(1), c[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    samples = flow.sample(v, (N_SAMPLES, 64), torch.zeros(N_SAMPLES, dtype=torch.long), steps=100).view(-1, 8, 8)
    rep(f"[2] Rectified flow, {FLOW_STEPS} steps, 100 Euler steps per sample ({time.time() - t0:.1f} s). Two samples:")
    rep(ascii_image(samples[0].clamp(0, 1)))
    rep(ascii_image(samples[1].clamp(0, 1)))
    results["rectified_flow"] = describe("rectified flow", samples, train, rep)
    rep()

    # 3. Raster autoregression over binary pixels.
    t0 = time.time()
    ar = TokenPrior(vocab=2, n_tokens=64, n_classes=1, order="raster", d=64, layers=2)
    pix = train.flatten(1).long()
    train_prior(ar, pix, c, steps=1500, batch=256)
    ar.eval(); ar.passes = 0
    s_ar = ar.sample_raster(torch.zeros(N_SAMPLES, dtype=torch.long)).float().view(-1, 8, 8)
    rep(f"[3] Raster autoregression, 1500 steps, {ar.passes} forward passes per batch of samples "
        f"({time.time() - t0:.1f} s). Two samples:")
    rep(ascii_image(s_ar[0]))
    rep(ascii_image(s_ar[1]))
    results["raster_autoregression"] = describe("raster autoregression", s_ar, train, rep)
    results["raster_autoregression"]["passes_per_image"] = ar.passes

    rep()
    rep("Committed = more than 90% of the image's brightness is in one half. Regression can never commit: its single "
        "output is the mean image, with half-bright blocks on both sides.")
    rep.data = results
    rep.save()


if __name__ == "__main__":
    main()
