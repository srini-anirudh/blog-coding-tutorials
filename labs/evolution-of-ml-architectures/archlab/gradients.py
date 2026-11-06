"""Products of Jacobians: how far a gradient survives through depth and through time.

The post's closing box says every family's failure is a product of Jacobians, and every fix is a change
to one factor in it. These helpers measure that product directly with autograd.
"""

from __future__ import annotations

import torch
from torch import nn


def deep_stack_input_grad(depth: int, width: int, act: str, residual: bool, seed: int = 0) -> float:
    """||dL/dx|| / ||dL/dy|| through `depth` Linear+activation layers at default init, with or without y = x + F(x)."""
    torch.manual_seed(seed)
    f = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu}[act]
    layers = [nn.Linear(width, width) for _ in range(depth)]
    x = torch.randn(64, width, requires_grad=True)
    h = x
    for lin in layers:
        h = h + f(lin(h)) if residual else f(lin(h))
    g = torch.randn_like(h)
    h.backward(g)
    return (x.grad.norm() / g.norm()).item()


def rnn_jacobian_norms(lam: float, T: int, d: int = 64, nonlinear: bool = True, seed: int = 0) -> torch.Tensor:
    """Spectral norm of dh_T/dh_t for t = T-1 ... 0, with W_h = lam * (random orthogonal), so sigma_max(W_h) = lam.

    Returns a tensor of length T: entry k is ||dh_T / dh_{T-k}||_2.
    """
    torch.manual_seed(seed)
    W = lam * torch.linalg.qr(torch.randn(d, d))[0]
    Wx = torch.randn(d, d) / d ** 0.5
    xs = 0.5 * torch.randn(T, d)
    h = torch.zeros(d)
    steps = []
    for t in range(T):
        a = W @ h + Wx @ xs[t]
        h = torch.tanh(a) if nonlinear else a
        steps.append((1 - h ** 2)[:, None] * W if nonlinear else W)  # dh_t/dh_{t-1} = diag(1 - h_t^2) W_h
    out = []
    J = torch.eye(d)
    for k in range(1, T + 1):  # J = dh_T/dh_{T-k} = steps[T-1] ... steps[T-k]
        J = J @ steps[T - k]
        out.append(torch.linalg.matrix_norm(J, ord=2).item())
    return torch.tensor(out)


def lstm_cell_path(forget_bias: float, T: int, d: int = 64, seed: int = 0) -> torch.Tensor:
    """||dc_T/dc_{T-k}|| along the cell state alone, = || prod diag(f_s) ||, for a random LSTM with that forget bias."""
    torch.manual_seed(seed)
    cell = nn.LSTMCell(d, d)
    with torch.no_grad():
        cell.bias_ih[d:2 * d] = forget_bias
        cell.bias_hh[d:2 * d] = 0.0
    h, c = torch.zeros(1, d), torch.zeros(1, d)
    fs = []
    with torch.no_grad():
        for _ in range(T):
            x = torch.randn(1, d)
            gates = x @ cell.weight_ih.T + cell.bias_ih + h @ cell.weight_hh.T + cell.bias_hh
            fs.append(torch.sigmoid(gates[0, d:2 * d]))
            h, c = cell(x, (h, c))
    f = torch.stack(fs)  # (T, d)
    prod = torch.cumprod(f.flip(0), 0)  # prod over the last k forget gates
    return prod.max(-1).values  # spectral norm of a diagonal matrix = its largest entry


def lstm_full_jacobian(forget_bias: float, ks: list[int], d: int = 64, seed: int = 0) -> list[float]:
    """||dc_T/dc_{T-k}||_2 through every path (cell and hidden), by autograd, for each distance k in ks."""
    torch.manual_seed(seed)
    cell = nn.LSTMCell(d, d)
    with torch.no_grad():
        cell.bias_ih[d:2 * d] = forget_bias
        cell.bias_hh[d:2 * d] = 0.0
    xs = torch.randn(max(ks), 1, d)
    h0 = torch.zeros(1, d)

    def run(c0, k):
        h, c = h0, c0
        for x in xs[:k]:
            h, c = cell(x, (h, c))
        return c

    c_start = 0.1 * torch.randn(1, d)
    out = []
    for k in ks:
        J = torch.autograd.functional.jacobian(lambda c0: run(c0, k), c_start)
        out.append(torch.linalg.matrix_norm(J.reshape(d, d), ord=2).item())
    return out


class Block(nn.Module):
    """A Transformer-shaped residual block with an MLP branch (enough to show where the norm sits)."""

    def __init__(self, d: int, pre: bool):
        super().__init__()
        self.pre = pre
        self.norm = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.f(self.norm(x)) if self.pre else self.norm(x + self.f(x))


def norm_placement_grads(depth: int, pre: bool, d: int = 64, seed: int = 0) -> list[float]:
    """Gradient norm of each block's weights at init under a random regression loss. Index 0 = bottom block."""
    torch.manual_seed(seed)
    blocks = nn.ModuleList(Block(d, pre) for _ in range(depth))
    x = torch.randn(32, 16, d)
    for b in blocks:
        x = b(x)
    loss = (x * torch.randn_like(x)).mean()
    loss.backward()
    return [torch.sqrt(sum((p.grad ** 2).sum() for p in b.f.parameters())).item() for b in blocks]
