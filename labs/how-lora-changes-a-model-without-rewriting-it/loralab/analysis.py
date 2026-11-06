"""Weight-space diagnostics from the post's DoRA and "PEFT versus full fine-tuning" sections.

magnitude_direction: DoRA's Delta M and Delta D (Liu et al. 2024, Sec. 3), per output row of an (out, in) weight.
intruder_count: Shuttleworth et al. (2024) intruder dimensions. A top singular vector of the fine-tuned matrix is an
intruder if its largest |cosine| with every singular vector of the pretrained matrix is below eps.
"""

from __future__ import annotations

import torch


def magnitude_direction(w0: torch.Tensor, w: torch.Tensor) -> tuple[float, float]:
    """Delta M = mean |m - m0|, Delta D = mean (1 - cos(w_row, w0_row)). Rows are output features."""
    w0, w = w0.float(), w.float()
    dm = (w.norm(dim=1) - w0.norm(dim=1)).abs().mean().item()
    dd = (1 - torch.nn.functional.cosine_similarity(w, w0, dim=1)).mean().item()
    return dm, dd


def intruder_count(w0: torch.Tensor, w: torch.Tensor, k: int = 10, eps: float = 0.5) -> int:
    """How many of w's top-k left singular vectors have max |cos| < eps with all left singular vectors of w0."""
    U0, _, _ = torch.linalg.svd(w0.float(), full_matrices=False)
    U, _, _ = torch.linalg.svd(w.float(), full_matrices=False)
    sims = (U[:, :k].T @ U0).abs().amax(dim=1)
    return int((sims < eps).sum().item())


def pearson(xs, ys) -> float:
    x, y = torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)
    x, y = x - x.mean(), y - y.mean()
    return (x @ y / (x.norm() * y.norm())).item()
