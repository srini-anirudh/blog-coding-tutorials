"""Two-dimensional loss surfaces from the post's first figure, plus a rotated bowl.

Each landscape has a loss f(xy), a start point, the minimum value f*, and optional gradient noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class Landscape:
    name: str
    start: tuple[float, float]
    f_star: float
    noise: float = 0.0

    def loss(self, xy: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Bowl(Landscape):
    """0.5 (x^2 + kappa y^2), optionally rotated by `angle` radians. Condition number kappa."""

    def __init__(self, name, kappa=29.0, angle=0.0, noise=0.0, start=(-4.0, 1.5)):
        super().__init__(name, start, 0.0, noise)
        self.kappa = kappa
        c, s = math.cos(angle), math.sin(angle)
        self.rot = torch.tensor([[c, -s], [s, c]])

    def loss(self, xy):
        x, y = self.rot.T @ xy
        return 0.5 * (x**2 + self.kappa * y**2)


class Rosenbrock(Landscape):
    """(1 - x)^2 + 100 (y - x^2)^2: a curved valley whose local curvature axes rotate along it."""

    def __init__(self):
        super().__init__("rosenbrock valley", (-1.5, 2.0), 0.0)

    def loss(self, xy):
        x, y = xy
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2


class SaddlePlateau(Landscape):
    """0.5 x^2 + 0.25 y^4 - 0.5 y^2: a saddle at the origin, minima at (0, +-1) with f* = -0.25.

    The start sits almost on the saddle, so the y gradient begins near 1e-3: a plateau.
    """

    def __init__(self):
        super().__init__("saddle + plateau", (2.0, 1e-3), -0.25)

    def loss(self, xy):
        x, y = xy
        return 0.5 * x**2 + 0.25 * y**4 - 0.5 * y**2


def landscapes() -> list[Landscape]:
    return [
        Bowl("ill-conditioned bowl, k=29"),
        Bowl("same bowl rotated 45 deg", angle=math.pi / 4),
        Rosenbrock(),
        SaddlePlateau(),
        Bowl("noisy bowl, k=29", noise=2.0),
    ]


def grad(land: Landscape, xy: torch.Tensor, gen: torch.Generator | None = None) -> tuple[torch.Tensor, float]:
    xy = xy.detach().requires_grad_(True)
    f = land.loss(xy)
    (g,) = torch.autograd.grad(f, xy)
    if land.noise:
        g = g + land.noise * torch.randn(2, generator=gen, dtype=g.dtype)
    return g, f.item()
