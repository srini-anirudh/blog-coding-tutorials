"""Plain deep MLPs with a chosen activation and initialisation, plus the probes sections 02 and 08 need.

`DeepMLP` keeps its linear layers in a list so a probe can read the gradient that reaches each one.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .activations import Activation, get


def init_linear(layer: nn.Linear, scheme: str, gain: float | None = None) -> None:
    """Initialise one layer's weight. Biases start at zero.

    xavier: Var(W) = gain^2 * 2 / (fan_in + fan_out)      (Glorot and Bengio 2010, gain 1 for tanh and sigmoid)
    he:     Var(W) = gain^2 / fan_in, gain sqrt(2)          (He et al. 2015, for the ReLU family)
    lecun:  Var(W) = 1 / fan_in                             (what SELU needs)
    """
    fan_out, fan_in = layer.weight.shape
    if scheme == "xavier":
        std = (gain or 1.0) * math.sqrt(2.0 / (fan_in + fan_out))
    elif scheme == "he":
        std = (gain or math.sqrt(2.0)) / math.sqrt(fan_in)
    elif scheme == "lecun":
        std = (gain or 1.0) / math.sqrt(fan_in)
    else:
        raise ValueError(f"unknown init scheme {scheme!r}")
    with torch.no_grad():
        layer.weight.normal_(0.0, std)
        if layer.bias is not None:
            layer.bias.zero_()


class DeepMLP(nn.Module):
    """in -> [Linear -> act] * depth -> Linear -> out. Every hidden layer has the same width."""

    def __init__(self, in_dim: int, width: int, out_dim: int, depth: int, act: str,
                 init: str = "he", gain: float | None = None) -> None:
        super().__init__()
        dims = [in_dim] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], out_dim)
        self.act = Activation(act)
        for layer in self.hidden:
            init_linear(layer, init, gain)
        init_linear(self.out, "xavier")

    def forward(self, x: torch.Tensor, return_hidden: bool = False, keep_pre: list | None = None):
        hs = []
        for layer in self.hidden:
            z = layer(x)
            if keep_pre is not None:
                z.retain_grad()
                keep_pre.append(z)
            x = self.act(z)
            hs.append(x)
        y = self.out(x)
        return (y, hs) if return_hidden else y


def grad_norms_per_layer(model: DeepMLP, x: torch.Tensor, y: torch.Tensor) -> dict[str, list[float]]:
    """One backward pass of cross entropy. Input side first, one entry per hidden layer.

    "signal": norm of dLoss/dz, the gradient arriving at each layer's pre-activation (the pulse in Fig 02).
    "weight": norm of dLoss/dW. It is the signal times the layer's input, so a forward signal that shrinks can
    hide a backward signal that shrinks (ReLU with Xavier does exactly this).
    """
    model.zero_grad(set_to_none=True)
    pre: list[torch.Tensor] = []
    loss = nn.functional.cross_entropy(model(x, keep_pre=pre), y)
    loss.backward()
    return {"signal": [z.grad.norm().item() for z in pre],
            "weight": [layer.weight.grad.norm().item() for layer in model.hidden]}


def decades_per_layer(norms: list[float]) -> float:
    """Least-squares slope of log10(norm) against layer index, read from the output side backwards.

    A negative value means the gradient shrinks as it travels toward the input.
    """
    logs = torch.tensor([math.log10(max(n, 1e-300)) for n in norms[::-1]], dtype=torch.float64)
    idx = torch.arange(len(logs), dtype=torch.float64)
    idx_c = idx - idx.mean()
    return float((idx_c * (logs - logs.mean())).sum() / (idx_c**2).sum())


@torch.no_grad()
def forward_second_moments(model: DeepMLP, x: torch.Tensor) -> list[float]:
    """E[h^2] of each hidden layer's output: does the signal shrink, hold, or grow with depth?"""
    _, hs = model(x, return_hidden=True)
    return [h.pow(2).mean().item() for h in hs]


def preserving_gain(act: str, q: float = 1.0, n: int = 2_000_000, seed: int = 0) -> float:
    """The weight gain g (Var(W) = g^2 / fan_in) that keeps the pre-activation second moment at q.

    With z ~ N(0, q), the next pre-activation has second moment g^2 * E[f(z)^2]. Setting it equal to q gives
    g = sqrt(q / E[f(z)^2]). For ReLU this is sqrt(2) at every q, because ReLU is positively homogeneous.
    For GELU and SiLU it depends on q, so "the" gain is only defined at a chosen scale.
    """
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, generator=gen, dtype=torch.float64) * math.sqrt(q)
    return math.sqrt(q / get(act)(z).pow(2).mean().item())
