"""Position: the only place geometry enters. RoPE and 2D RoPE are imported from C3 by the scripts.

sinusoidal              the 2017 fixed table, one sin/cos pair per frequency
sincos_2d               half the channels encode the row, half the column
interpolate_table       the standard fine-tune step: a learned 1D table -> grid -> bicubic -> new grid
neighbour_score         did a learned 1D table recover 2D structure? (closest row is a grid neighbour)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sinusoidal(n: int, dim: int, base: float = 10_000.0) -> torch.Tensor:
    """(n, dim). Columns 2i and 2i+1 are sin and cos of position * base^(-2i/dim)."""
    pos = torch.arange(n, dtype=torch.float64)[:, None]
    freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    table = torch.zeros(n, dim, dtype=torch.float64)
    table[:, 0::2] = torch.sin(pos * freq)
    table[:, 1::2] = torch.cos(pos * freq)
    return table.float()


def sinusoidal_shift_matrix(k: int, dim: int, base: float = 10_000.0) -> torch.Tensor:
    """The fixed matrix M_k with sinusoidal(t + k) = sinusoidal(t) @ M_k for every t: a 2x2 rotation per pair."""
    freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    m = torch.zeros(dim, dim, dtype=torch.float64)
    c, s = torch.cos(k * freq), torch.sin(k * freq)
    i = torch.arange(0, dim, 2)
    # [sin(a+b), cos(a+b)] = [sin a, cos a] @ [[cos b, -sin b], [sin b, cos b]]
    m[i, i], m[i, i + 1], m[i + 1, i], m[i + 1, i + 1] = c, -s, s, c
    return m.float()


def sincos_2d(height: int, width: int, dim: int) -> torch.Tensor:
    """(height*width, dim) in raster order: first half from the row index, second half from the column index."""
    rows = sinusoidal(height, dim // 2)
    cols = sinusoidal(width, dim // 2)
    r = rows[:, None, :].expand(height, width, dim // 2)
    c = cols[None, :, :].expand(height, width, dim // 2)
    return torch.cat([r, c], dim=-1).reshape(height * width, dim)


def interpolate_table(table: torch.Tensor, old_hw: tuple[int, int], new_hw: tuple[int, int]) -> torch.Tensor:
    """(h*w, d) learned 1D table -> (h'*w', d): reshape to a grid, resize bicubically, flatten again."""
    (h, w), d = old_hw, table.shape[-1]
    grid = table.reshape(1, h, w, d).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=new_hw, mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(new_hw[0] * new_hw[1], d)


def cosine_matrix(table: torch.Tensor) -> torch.Tensor:
    t = F.normalize(table.float(), dim=-1)
    return t @ t.T


def neighbour_score(table: torch.Tensor, height: int, width: int) -> float:
    """Fraction of grid positions whose most similar other row is one of its 4 grid neighbours.

    A random table scores about 4 / (h*w - 1). A table that knows the 2D layout scores near 1.
    """
    sim = cosine_matrix(table)
    sim.fill_diagonal_(-2.0)
    best = sim.argmax(dim=-1)
    r, c = torch.arange(height * width) // width, torch.arange(height * width) % width
    dist = (r - r[best]).abs() + (c - c[best]).abs()
    return (dist == 1).float().mean().item()


def similarity_by_distance(table: torch.Tensor, height: int, width: int, max_dist: int = 6) -> list[float]:
    """Mean cosine similarity between rows at grid (Manhattan) distance 1..max_dist."""
    sim = cosine_matrix(table)
    r, c = torch.arange(height * width) // width, torch.arange(height * width) % width
    dist = (r[:, None] - r[None]).abs() + (c[:, None] - c[None]).abs()
    return [sim[dist == k].mean().item() for k in range(1, max_dist + 1)]


def alignment_score(old: torch.Tensor, new: torch.Tensor, old_hw: tuple[int, int], new_hw: tuple[int, int]) -> float:
    """After resizing a table, does each new position look most like the old position it now covers?

    The centre of new cell (i, j) falls in old cell ((i + 0.5) h / h', (j + 0.5) w / w'), rounded down. Score = fraction of new cells whose most similar
    old row is that one. Interpolation keeps the layout; reusing raster rows does not.
    """
    (h, w), (nh, nw) = old_hw, new_hw
    sim = F.normalize(new.float(), dim=-1) @ F.normalize(old.float(), dim=-1).T  # (nh*nw, h*w)
    i, j = torch.arange(nh * nw) // nw, torch.arange(nh * nw) % nw
    target = ((2 * i + 1) * h // (2 * nh)) * w + (2 * j + 1) * w // (2 * nw)
    return (sim.argmax(dim=-1) == target).float().mean().item()
