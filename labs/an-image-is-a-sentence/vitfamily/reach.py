"""Who can reach whom. One block is an N x N boolean matrix A: A[i, j] = token i reads token j.

Depth composes by boolean matrix product, so reach after L blocks is a one-hot row times A_1 ... A_L.
Swin's shifted windows use C3's cyclic shift and region mask, so the matrix is the one the real
SW-MSA block computes, wrap-around seam included.
"""

from __future__ import annotations

import torch

from .c03 import shifted_window_mask, window_partition


def grid_index(side: int) -> torch.Tensor:
    return torch.arange(side * side).view(side, side)


def global_block(side: int) -> torch.Tensor:
    return torch.ones(side * side, side * side, dtype=torch.bool)


def window_block(side: int, m: int, shift: int = 0, masked: bool = True) -> torch.Tensor:
    """W-MSA (shift 0) or SW-MSA (shift > 0) connectivity. masked=False drops Swin's mask: the torus bug."""
    idx = torch.roll(grid_index(side), shifts=(-shift, -shift), dims=(0, 1))
    ids = window_partition(idx[None, :, :, None], m).squeeze(-1)  # (nW, M^2) original token ids
    if shift and masked:
        allowed = shifted_window_mask(side, side, m, shift)  # (nW, M^2, M^2)
    else:
        allowed = torch.ones(ids.size(0), m * m, m * m, dtype=torch.bool)
    a = torch.zeros(side * side, side * side, dtype=torch.bool)
    rows = ids[:, :, None].expand_as(allowed)
    cols = ids[:, None, :].expand_as(allowed)
    a[rows[allowed], cols[allowed]] = True
    return a


def axial_blocks(side: int) -> list[torch.Tensor]:
    """Row pass, then column pass."""
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    y, x = y.flatten(), x.flatten()
    return [y[:, None] == y[None, :], x[:, None] == x[None, :]]


def conv_block(side: int, k: int = 3) -> torch.Tensor:
    """A k x k convolution with zero padding: no wrap-around."""
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    y, x = y.flatten(), x.flatten()
    r = k // 2
    return ((y[:, None] - y[None, :]).abs() <= r) & ((x[:, None] - x[None, :]).abs() <= r)


def reach_over_depth(blocks: list[torch.Tensor], start: int) -> list[torch.Tensor]:
    """Boolean reach set (N,) of token `start` after each block. Residuals keep the token itself."""
    r = torch.zeros(blocks[0].size(0), dtype=torch.bool)
    r[start] = True
    out = []
    for a in blocks:
        r = (r.float() @ a.float() > 0) | r
        out.append(r.clone())
    return out


def chebyshev_radius(reach: torch.Tensor, side: int, start: int) -> int:
    """Largest max(|dy|, |dx|) from `start` among reached tokens."""
    y, x = divmod(start, side)
    idx = reach.nonzero().squeeze(1)
    return int(torch.maximum((idx // side - y).abs(), (idx % side - x).abs()).max())


def blocks_to_cover(blocks: list[torch.Tensor], start: int) -> int | None:
    """First depth (1-based) at which `start` reaches every token, or None."""
    for depth, r in enumerate(reach_over_depth(blocks, start), start=1):
        if bool(r.all()):
            return depth
    return None


# ---- space-time schemes on a (T, S, S) token grid ----------------------------------------------
def st_passes(scheme: str, t: int, side: int) -> list[torch.Tensor]:
    """The attention passes of one block, as boolean (T N, T N) matrices. TimeSformer's naming."""
    f, y, x = torch.meshgrid(torch.arange(t), torch.arange(side), torch.arange(side), indexing="ij")
    f, y, x = f.flatten(), y.flatten(), x.flatten()
    same_f, same_y, same_x = f[:, None] == f[None, :], y[:, None] == y[None, :], x[:, None] == x[None, :]
    same_patch = same_y & same_x
    if scheme == "S":
        return [same_f]
    if scheme == "ST":
        return [torch.ones_like(same_f)]
    if scheme == "T+S":
        return [same_patch, same_f]
    if scheme == "T+W+H":
        return [same_patch, same_f & same_y, same_f & same_x]
    raise ValueError(f"unknown scheme {scheme!r}")


def keys_per_query(passes: list[torch.Tensor]) -> int:
    """Scores one query computes in one block, summed over the block's passes."""
    return sum(int(p[0].sum()) for p in passes)


def reach_after_block(passes: list[torch.Tensor], start: int = 0) -> int:
    return int(reach_over_depth(passes, start)[-1].sum())
