"""Decoupled weight decay sets an equilibrium weight norm, RMS(theta) ~ sqrt(lr / (2 wd)).

Claims checked:
  1. The recursion theta <- (1 - lr wd) theta - lr u, with u uncorrelated and RMS(u) = 1, settles at
     RMS(theta) = sqrt(lr / (2 wd)). For lr = 3e-4 and wd = 0.1 that is about 0.04.
  2. A real AdamW run on a scale-invariant layer (a linear map followed by a norm) settles near the same
     scale, multiplied by the RMS of Adam's actual update, which is below 1 when gradients are noisy.
  3. Halving lr moves the equilibrium by sqrt(1/2), not by 1/2.
  4. With a cosine schedule, the decay term lr(t) wd almost vanishes in the last 10% of training.

Run: python labs/what-an-optimizer-actually-does/scripts/04_weight_decay.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optlab import Adam, rms
from optlab.common import Report


def recursion(lr, wd, dim=4096, seed=0):
    """Run 8 time constants, average RMS(theta) over the last 25% of steps."""
    gen = torch.Generator().manual_seed(seed)
    theta = torch.zeros(dim)
    steps = int(8 / (lr * wd))
    acc, n = 0.0, 0
    for t in range(steps):
        theta.mul_(1 - lr * wd).add_(torch.randn(dim, generator=gen), alpha=-lr)
        if t > 0.75 * steps and t % 10 == 0:
            acc, n = acc + rms(theta).item(), n + 1
    return acc / n


def scale_invariant_run(lr, wd, steps=6000, seed=0, beta1=0.9):
    """Train W (64 x 32) inside y = readout . normalize(W x) to fit noisy targets with AdamW.

    The loss does not change if W is multiplied by a constant, so only decay and the update size set |W|.
    """
    torch.manual_seed(seed)
    w_true = torch.randn(64, 32)
    readout = torch.randn(64) / 8
    W = torch.nn.Parameter(torch.randn(64, 32) * 0.02)
    opt = Adam([W], lr=lr, betas=(beta1, 0.999), weight_decay=wd)
    w_rms, u_rms = [], []
    for t in range(steps):
        x = torch.randn(128, 32)
        h = x @ w_true.T
        target = (h / h.norm(dim=1, keepdim=True)) @ readout + 0.05 * torch.randn(128)
        z = x @ W.T
        pred = (z / z.norm(dim=1, keepdim=True)) @ readout
        loss = (pred - target).pow(2).mean()
        W.grad = None
        loss.backward()
        before = W.detach().clone()
        opt.step()
        if t > steps // 2:
            w_rms.append(rms(W).item())
            # The Adam part of the step, with the decay term removed, divided by lr.
            u_rms.append(rms((W.detach() - before * (1 - lr * wd)) / lr).item())
    return sum(w_rms) / len(w_rms), sum(u_rms) / len(u_rms), loss.item()


def main():
    rep = Report("04_weight_decay")
    rep("## 1. The idealized recursion, u ~ N(0, 1) i.i.d., 4096 coordinates")
    rep(f"{'lr':>8} {'wd':>5} {'measured RMS':>13} {'sqrt(lr/2wd)':>13}")
    ideal = {}
    for lr, wd in ((3e-4, 0.1), (1e-3, 0.1), (1e-2, 0.1), (1e-3, 1.0), (5e-4, 0.1)):
        r = recursion(lr, wd)
        ideal[f"{lr}/{wd}"] = r
        rep(f"{lr:>8.0e} {wd:>5} {r:>13.4f} {math.sqrt(lr / (2 * wd)):>13.4f}")
    rep(f"halving lr from 1e-3 to 5e-4 at wd 0.1: ratio {ideal['0.0005/0.1'] / ideal['0.001/0.1']:.3f} "
        f"(sqrt(1/2) = {math.sqrt(0.5):.3f})")

    rep("\n## 2. AdamW on a scale-invariant layer, 6000 steps, averages over the second half")
    rep("RMS(u) is the per-step RMS of Adam's direction m_hat/(sqrt(v_hat)+eps), decay term removed")
    rep(f"{'b1':>4} {'lr':>6} {'wd':>5} {'RMS(W)':>8} {'sqrt(lr/2wd)':>13} {'ratio':>6} {'RMS(u)':>7} {'loss':>8}")
    real = {}
    for b1 in (0.9, 0.0):
        for lr, wd in ((1e-2, 0.1), (1e-2, 0.5), (5e-3, 0.5), (1e-2, 2.0)):
            w, u, loss = scale_invariant_run(lr, wd, beta1=b1)
            pred = math.sqrt(lr / (2 * wd))
            real[f"b1={b1} {lr}/{wd}"] = dict(rms_w=w, rms_u=u, formula=pred)
            rep(f"{b1:>4} {lr:>6.0e} {wd:>5} {w:>8.4f} {pred:>13.4f} {w / pred:>6.2f} {u:>7.3f} {loss:>8.4f}")

    rep("\n## 3. Cosine schedule from peak to 0: share of the total decay sum(lr(t) wd) applied in each tenth")
    T = 10_000
    lr_t = [0.5 * (1 + math.cos(math.pi * t / T)) for t in range(T)]
    total = sum(lr_t)
    tenths = [sum(lr_t[i * T // 10 : (i + 1) * T // 10]) / total for i in range(10)]
    rep(" ".join(f"{x:.3f}" for x in tenths))
    rep(f"last tenth: {tenths[-1]:.2%} of all decay; first tenth: {tenths[0]:.2%}")
    rep.save({"ideal": ideal, "adamw_scale_invariant": real, "cosine_tenths": tenths})


if __name__ == "__main__":
    main()
