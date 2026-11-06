"""Broadcasting rules, written out by hand.

Two shapes are compared from the right. A missing dim counts as size 1.
Sizes must be equal, or one of them must be 1; a size-1 dim is stretched
with stride 0, so no data is copied.
"""

from __future__ import annotations

import torch


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """The result shape of `a op b`, or ValueError when the shapes do not broadcast."""
    rank = max(len(a), len(b))
    # Left-pad the shorter shape with 1s so both have the same rank.
    a_pad = (1,) * (rank - len(a)) + tuple(a)
    b_pad = (1,) * (rank - len(b)) + tuple(b)
    out = []
    for dim, (x, y) in enumerate(zip(a_pad, b_pad)):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"dim {dim - rank}: sizes {x} and {y} do not broadcast (shapes {a} and {b})")
    return tuple(out)


def expand_to(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Broadcast `t` to `shape` by setting stride 0 on every stretched dim."""
    rank = len(shape)
    src_shape = (1,) * (rank - t.dim()) + tuple(t.shape)
    src_stride = (0,) * (rank - t.dim()) + tuple(t.stride())
    stride = []
    for size, src_size, src_st in zip(shape, src_shape, src_stride):
        if src_size == size:
            stride.append(src_st)
        elif src_size == 1:
            stride.append(0)  # every index along this dim reads the same element
        else:
            raise ValueError(f"cannot expand {tuple(t.shape)} to {shape}")
    return t.as_strided(shape, stride, t.storage_offset())
