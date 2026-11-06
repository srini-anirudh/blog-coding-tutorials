"""Lazy and rich training of a two-layer ReLU network.

The network is in the NTK parameterization, f(x) = (1/sqrt(m)) sum_j a_j relu(w_j . x), with a_j ~ N(0, 1)
and w_j ~ N(0, I/d). Following Chizat, Oyallon and Bach (2019), we train the centred, scaled model
    g(x) = alpha * (f(x; theta) - f(x; theta_0))
on the loss (1 / alpha^2) * 0.5 * mean((g - y)^2) with plain full-batch gradient descent. To first order the
function-space dynamics do not depend on alpha. What alpha controls is how far theta must move to get
there: large alpha means tiny moves (lazy, the network stays at its linearization), small alpha means
large moves (rich, the hidden features turn towards the task).

We measure three things per run: test error, how far the first-layer weights moved, and how much the
tangent kernel K(x, x') = <grad_theta f(x), grad_theta f(x')> changed. We also solve the lazy limit
exactly: kernel regression with the initial tangent kernel, which is what the linearized model converges to.
"""

import math

import torch


def teacher_data(n, d=10, seed=0):
    """A single-index target y = relu(u . x) - E[relu(u . x)]: it depends on one direction u in R^d,
    which a feature-learning network can find and a fixed kernel cannot."""
    g = torch.Generator().manual_seed(seed)
    u = torch.zeros(d)
    u[0] = 1.0
    x = torch.randn(n, d, generator=g)
    y = torch.relu(x @ u) - 1.0 / math.sqrt(2 * math.pi)
    return x, y, u


class TwoLayer(torch.nn.Module):
    def __init__(self, d, m, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(m, d, generator=g) / math.sqrt(d))
        self.a = torch.nn.Parameter(torch.randn(m, generator=g))

    def forward(self, x):
        return torch.relu(x @ self.w.T) @ self.a / math.sqrt(self.a.numel())


def tangent_features(model, x):
    """The n x P matrix of per-example gradients df(x_i)/dtheta, in closed form for this network."""
    m = model.a.numel()
    pre = x @ model.w.T                                   # n x m
    d_a = torch.relu(pre) / math.sqrt(m)                  # df/da_j
    d_w = ((pre > 0).float() * model.a / math.sqrt(m))[:, :, None] * x[:, None, :]   # df/dw_j
    return torch.cat([d_a, d_w.flatten(1)], dim=1)


def kernel_distance(K0, K1):
    """1 - cosine similarity between two kernel matrices (0 means unchanged)."""
    return float(1 - (K0 * K1).sum() / (K0.norm() * K1.norm()))


def train(alpha, m=1024, n=200, d=10, steps=4000, lr=1.0, seed=0, n_test=2000):
    """Train g = alpha (f - f0) with gradient descent. Returns the metrics described in the module doc."""
    x, y, _ = teacher_data(n, d, seed)
    xt, yt, _ = teacher_data(n_test, d, seed + 1000)
    model = TwoLayer(d, m, seed)
    w0 = model.w.detach().clone()
    with torch.no_grad():
        f0, f0t = model(x), model(xt)
    phi0 = tangent_features(model, x).detach()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        g = alpha * (model(x) - f0)
        loss = 0.5 * ((g - y) ** 2).mean() / alpha**2
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_mse = float(((alpha * (model(x) - f0) - y) ** 2).mean())
        test_mse = float(((alpha * (model(xt) - f0t) - yt) ** 2).mean())
        phi1 = tangent_features(model, x)
        dw = model.w - w0
    return {
        "alpha": alpha, "width": m, "train_mse": train_mse, "test_mse": test_mse,
        "rel_weight_change": float(dw.norm() / w0.norm()),
        "kernel_change": kernel_distance(phi0 @ phi0.T, phi1 @ phi1.T),
    }


def lazy_limit(m=1024, n=200, d=10, seed=0, n_test=2000):
    """Test MSE of kernel interpolation with the initial tangent kernel: where training goes as alpha -> inf."""
    x, y, _ = teacher_data(n, d, seed)
    xt, yt, _ = teacher_data(n_test, d, seed + 1000)
    model = TwoLayer(d, m, seed).requires_grad_(False)
    phi, phit = tangent_features(model, x).double(), tangent_features(model, xt).double()
    coef = torch.linalg.solve(phi @ phi.T, y.double())
    return float(((phit @ phi.T @ coef - yt.double()) ** 2).mean())
