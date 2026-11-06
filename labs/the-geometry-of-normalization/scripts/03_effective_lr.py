"""Section 01, story 3: scale invariance turns the weight norm into a learning-rate knob.

Model: logits = V @ LayerNorm(W @ x), with LayerNorm without gain or bias, and V fixed. Only W trains.
The forward pass cannot see ||W||, so:
  forward   f(kW) = f(W)
  backward  grad(kW) = grad(W) / k, and grad(W) is orthogonal to W
  SGD step  angle moved per step ~ lr * ||grad|| / ||W||, which scales as 1/k^2
Without weight decay ||W||^2 grows by lr^2 ||grad||^2 every step, so the angular step decays by itself.
With weight decay lambda, ||W|| settles where the shrink balances the growth. There, setting
(1 - lr*lambda)^2 ||W||^2 + lr^2 ||grad||^2 = ||W||^2 gives an angular step of about sqrt(2 * lr * lambda).
"""

import argparse
import math

import _setup  # noqa: F401
import torch
import torch.nn.functional as F

from normgeo.report import Report

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--lr", type=float, default=0.2)
args = p.parse_args()

torch.manual_seed(0)
D_IN, D_H, K, N = 32, 16, 4, 4096
X = torch.randn(N, D_IN)
teacher = torch.randn(K, D_IN)
Y = (X @ teacher.T + 0.5 * torch.randn(N, K)).argmax(-1)  # a noisy linear teacher: not perfectly separable
V = torch.randn(K, D_H) / math.sqrt(D_H)
W0 = torch.randn(D_H, D_IN) / math.sqrt(D_IN)


def loss_fn(W: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(F.layer_norm(x @ W.T, (D_H,)) @ V.T, y)


def grad(W: torch.Tensor, x=X, y=Y) -> torch.Tensor:
    W = W.detach().requires_grad_(True)
    loss_fn(W, x, y).backward()
    return W.grad


def angle(a: torch.Tensor, b: torch.Tensor) -> float:
    return math.acos(min(1.0, F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()))


with Report("03_effective_lr") as say:
    say(f"Part A. Rescale W by k. One full-batch SGD step at lr {args.lr}.")
    say(f"{'k':>6}{'loss':>10}{'||grad||':>12}{'k*||grad||':>12}{'angle (rad)':>14}{'k^2*angle':>12}{'<grad,W>/(|g||W|)':>20}")
    for k in [0.25, 0.5, 1.0, 2.0, 4.0]:
        W = W0 * k
        g = grad(W)
        a = angle(W, W - args.lr * g)
        cos_gw = F.cosine_similarity(g.flatten(), W.flatten(), dim=0).item()
        say(f"{k:>6}{loss_fn(W, X, Y).item():>10.4f}{g.norm():>12.4f}{k * g.norm():>12.4f}{a:>14.2e}{k * k * a:>12.2e}{cos_gw:>20.1e}")

    say()
    say(f"Part B. Train W with minibatch SGD (batch 64, lr {args.lr}) for {args.steps} steps, three weight decays.")
    say("angle = mean angular step over the first or last 200 steps. sqrt(2 lr wd) is the predicted equilibrium angle.")
    say(f"{'wd':>8}{'||W|| start':>13}{'||W|| end':>11}{'angle first':>13}{'angle last':>12}{'sqrt(2 lr wd)':>15}{'loss end':>10}{'max growth-law error':>22}")
    curves = {}
    for wd in [0.0, 1e-3, 1e-2]:
        torch.manual_seed(1)
        W, angles, norms, worst = W0.clone(), [], [], 0.0
        for step in range(args.steps):
            idx = torch.randint(0, N, (64,))
            g = grad(W, X[idx], Y[idx])
            new = W - args.lr * (g + wd * W)
            if wd == 0.0:  # check ||W'||^2 = ||W||^2 + lr^2 ||g||^2 exactly (g is orthogonal to W)
                pred = W.norm() ** 2 + args.lr**2 * g.norm() ** 2
                worst = max(worst, abs(new.norm().item() ** 2 - pred.item()) / pred.item())
            angles.append(angle(W, new))
            W = new
            norms.append(W.norm().item())
        tail = sum(angles[-200:]) / 200
        pred_angle = f"{math.sqrt(2 * args.lr * wd):.2e}" if wd else "-"
        err = f"{worst:.1e}" if wd == 0.0 else "-"
        say(f"{wd:>8}{W0.norm():>13.3f}{norms[-1]:>11.3f}{sum(angles[:200]) / 200:>13.2e}{tail:>12.2e}{pred_angle:>15}{loss_fn(W, X, Y).item():>10.4f}{err:>22}")
        curves[str(wd)] = {"norm": norms[::50], "angle": [sum(angles[i : i + 50]) / 50 for i in range(0, args.steps, 50)]}
    say()
    say("With wd = 0 the norm only grows, and the angular step falls with it: an automatic learning-rate decay.")
    say("With wd > 0 the norm stops growing and the angular step settles near sqrt(2 lr wd).")
    say.json({"lr": args.lr, "steps": args.steps, "every": 50, "curves": curves})
