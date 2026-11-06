"""Parameter space is not function space: permutation and rescaling symmetries, and what they do to sharpness.

A two-layer ReLU network f(x) = W2 relu(W1 x + b1) + b2 computes the same function after
  * permuting its hidden units (rows of W1, entries of b1, columns of W2), and
  * multiplying (W1, b1) by c > 0 and dividing W2 by c, since relu(c z) = c relu(z).
Dinh et al. (2017) point out that the second one changes the Hessian of the loss, so "sharpness" measured in
raw parameter coordinates can be set to anything while the function stays fixed. We measure it.
"""

import math

import torch
from torch import nn


def small_mlp(width=64, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, 1))


def regression_data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 2 - 1
    y = torch.sin(2.3 * x) + 0.42 * x**2 + 0.1 * torch.randn(n, 1, generator=g)
    return x, y


def train(model, x, y, steps=3000, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    return float(loss.detach())


@torch.no_grad()
def permuted(model, perm):
    """A copy of `model` with its hidden units reordered by `perm`."""
    new = small_mlp(model[0].out_features)
    new.load_state_dict(model.state_dict())
    new[0].weight.copy_(model[0].weight[perm])
    new[0].bias.copy_(model[0].bias[perm])
    new[2].weight.copy_(model[2].weight[:, perm])
    return new


@torch.no_grad()
def rescaled(model, c):
    """A copy of `model` with layer 1 multiplied by c and layer 2's weights divided by c."""
    new = small_mlp(model[0].out_features)
    new.load_state_dict(model.state_dict())
    new[0].weight.mul_(c)
    new[0].bias.mul_(c)
    new[2].weight.div_(c)
    return new


def top_hessian_eigenvalue(model, x, y, iters=100, seed=0):
    """Largest eigenvalue of the Hessian of the training MSE with respect to all parameters,
    by power iteration on Hessian-vector products (double backward)."""
    params = [p for p in model.parameters()]
    loss = nn.functional.mse_loss(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=g) for p in params]
    eig = 0.0
    for _ in range(iters):
        norm = torch.sqrt(sum((vi**2).sum() for vi in v))
        v = [vi / norm for vi in v]
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        eig = float(sum((h * vi).sum() for h, vi in zip(hv, v)))
        v = [h.detach() for h in hv]
    return eig


def param_norm(model):
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())))


def log10_factorial(m):
    """log10(m!), the number of hidden-unit orderings of one width-m layer."""
    return math.lgamma(m + 1) / math.log(10)
