"""Random Fourier features (Tancik et al., 2020): gamma(v) = [cos(2 pi B v), sin(2 pi B v)], B ~ N(0, sigma^2).

Two uses from the post:
  a coordinate MLP that can fit high-frequency detail only after this mapping, with sigma as the knob;
  SAM-style prompt encoding, where one shared B makes "click vs grid cell" a smooth similarity kernel.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def gaussian_B(n_feat: int, in_dim: int, sigma: float, seed: int = 0) -> torch.Tensor:
    return torch.randn(n_feat, in_dim, generator=torch.Generator().manual_seed(seed)) * sigma


def encode(v: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """v (N, in_dim) -> (N, 2 * n_feat)."""
    proj = 2 * math.pi * v @ B.T
    return torch.cat([proj.cos(), proj.sin()], -1)


def kernel(v1: torch.Tensor, v2: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """gamma(v1) . gamma(v2) / n_feat = mean_j cos(2 pi b_j . (v1 - v2)). Depends only on v1 - v2."""
    return encode(v1, B) @ encode(v2, B).T / B.shape[0]


def expected_kernel(delta: torch.Tensor, sigma: float) -> torch.Tensor:
    """E[cos(2 pi b . delta)] for b ~ N(0, sigma^2 I) = exp(-2 pi^2 sigma^2 |delta|^2): a Gaussian of width ~ 1/sigma."""
    return torch.exp(-2 * math.pi**2 * sigma**2 * (delta**2).sum(-1))


def grid(n: int) -> torch.Tensor:
    """(n*n, 2) cell-centre coordinates in [0, 1], row-major (y, x)."""
    c = (torch.arange(n) + 0.5) / n
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([yy, xx], -1).reshape(-1, 2)


def test_image(n: int = 128) -> torch.Tensor:
    """(n, n) a deterministic image with coarse shapes and fine stripes, values in [0, 1]."""
    c = torch.arange(n, dtype=torch.float32) / n
    y, x = torch.meshgrid(c, c, indexing="ij")
    img = 0.5 + 0.25 * torch.sin(2 * math.pi * (3 * x + 2 * y))
    img = img + 0.2 * (((x - 0.3) ** 2 + (y - 0.35) ** 2) < 0.04).float()
    img = img + 0.15 * torch.sign(torch.sin(2 * math.pi * 14 * x)) * (y > 0.6).float()
    img = img + 0.1 * torch.sin(2 * math.pi * 21 * (x * 0.6 + y * 0.8)) * (x > 0.55).float()
    return img.clamp(0, 1)


class CoordMLP(nn.Module):
    """Coordinates -> value. `B=None` means raw coordinates (the blurry baseline)."""

    def __init__(self, B: torch.Tensor | None, width: int = 256, depth: int = 3) -> None:
        super().__init__()
        self.register_buffer("B", B if B is not None else torch.empty(0), persistent=False)
        d_in = 2 if B is None else 2 * B.shape[0]
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v if self.B.numel() == 0 else encode(v, self.B)).squeeze(-1)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(((pred.clamp(0, 1) - target) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def fit_image(img: torch.Tensor, B: torch.Tensor | None, steps: int = 2000, lr: float = 1e-3, seed: int = 0):
    """Train on the even pixels, test on the rest. Returns (train_psnr, test_psnr, prediction (n, n))."""
    torch.manual_seed(seed)
    n = img.shape[0]
    coords = grid(n)
    values = img.reshape(-1)
    idx = torch.arange(n * n).reshape(n, n)
    train = idx[0::2, 0::2].reshape(-1)
    test = torch.ones(n * n, dtype=torch.bool)
    test[train] = False
    model = CoordMLP(B)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = ((model(coords[train]) - values[train]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(coords)
    return psnr(pred[train], values[train]), psnr(pred[test], values[test]), pred.reshape(n, n)
