"""Gradient checkpointing from scratch: keep k segment boundaries, rebuild each segment during backward.

`backprop_segments` runs n layers as k segments. The forward pass runs under `no_grad` and keeps only the input
of each segment. The backward pass walks the segments from last to first. For each one it re-runs the forward
pass with the graph on, backpropagates the incoming gradient through it, and frees it before moving on.

With k = 1 the one segment is the whole network, which is ordinary backpropagation plus one wasted forward pass;
`plain_backprop` is the real baseline. Every layer evaluation is counted, so the compute side of the trade is a
number too.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .saved import SavedTensorLedger


class Block(nn.Module):
    """One residual MLP layer. All layers cost the same to store and to recompute, as the sqrt(n) analysis assumes."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, width)
        self.evals = 0

    def forward(self, x):
        self.evals += 1
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x))))


def make_chain(n_layers: int, width: int, seed: int = 0) -> nn.ModuleList:
    torch.manual_seed(seed)
    return nn.ModuleList(Block(width) for _ in range(n_layers))


def _reset(layers):
    for layer in layers:
        layer.evals = 0
        for p in layer.parameters():
            p.grad = None


def plain_backprop(layers, x) -> dict:
    """Ordinary reverse mode: every activation is kept until backward reaches it."""
    _reset(layers)
    params = [p for layer in layers for p in layer.parameters()]
    with SavedTensorLedger(exclude=params) as ledger:
        h = x
        for layer in layers:
            h = layer(h)
        loss = h.pow(2).mean()
    loss.backward()
    return dict(loss=loss.item(), peak_bytes=ledger.peak, layer_evals=sum(lay.evals for lay in layers))


def segment_bounds(n: int, k: int) -> list[tuple[int, int]]:
    """Split range(n) into k contiguous segments as evenly as possible."""
    edges = [round(i * n / k) for i in range(k + 1)]
    return [(edges[i], edges[i + 1]) for i in range(k) if edges[i] < edges[i + 1]]


def backprop_segments(layers, x, k: int) -> dict:
    _reset(layers)
    segs = segment_bounds(len(layers), k)
    params = [p for layer in layers for p in layer.parameters()]
    with SavedTensorLedger(exclude=params) as ledger:
        # Forward: keep only the boundaries.
        boundaries = []
        h = x
        with torch.no_grad():
            for a, b in segs:
                boundaries.append(h)
                ledger.add_extra(h.untyped_storage().nbytes())
                for layer in layers[a:b]:
                    h = layer(h)
        # The loss on the final output is cheap; build its graph directly.
        out = h.detach().requires_grad_(True)
        loss = out.pow(2).mean()
        loss.backward()
        grad = out.grad
        # Backward: rebuild one segment at a time, from the last to the first.
        for (a, b), start in zip(reversed(segs), reversed(boundaries)):
            start = start.detach().requires_grad_(True)
            h = start
            for layer in layers[a:b]:
                h = layer(h)
            h.backward(grad)
            grad = start.grad
            ledger.add_extra(-start.untyped_storage().nbytes())
            del h, start
    return dict(loss=loss.item(), peak_bytes=ledger.peak, layer_evals=sum(lay.evals for lay in layers))


def best_k(n: int) -> int:
    """The k that minimises the segment-count model k + n/k: about sqrt(n)."""
    return min(range(1, n + 1), key=lambda k: k + math.ceil(n / k))


def grads_of(layers) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for layer in layers for p in layer.parameters()])
