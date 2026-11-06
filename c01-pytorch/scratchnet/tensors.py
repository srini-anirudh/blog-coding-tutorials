"""Tools for looking inside a tensor: storage, strides, offsets and views.

A tensor is a view: a flat block of memory (the storage) plus metadata
(shape, stride, storage offset, dtype, device) that says how to read it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Layout:
    """Everything PyTorch needs to turn a logical index into a memory address."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int
    dtype: torch.dtype
    device: str
    element_size: int
    nbytes: int  # bytes this tensor's elements occupy: numel * element_size
    storage_bytes: int  # bytes of the whole storage it points into
    contiguous: bool


def memory_bytes(t: torch.Tensor) -> int:
    """M = numel(t) * bytes-per-element."""
    return t.numel() * t.element_size()


def describe(t: torch.Tensor) -> Layout:
    """Collect the layout metadata of `t` in one record."""
    return Layout(
        shape=tuple(t.shape),
        stride=tuple(t.stride()),
        offset=t.storage_offset(),
        dtype=t.dtype,
        device=str(t.device),
        element_size=t.element_size(),
        nbytes=memory_bytes(t),
        storage_bytes=t.untyped_storage().nbytes(),
        contiguous=t.is_contiguous(),
    )


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major strides: the last dim moves by 1, each earlier dim by the product of later sizes."""
    strides = []
    step = 1
    for size in reversed(shape):
        strides.append(step)
        step *= size
    return tuple(reversed(strides))


def storage_index(index: tuple[int, ...], stride: tuple[int, ...], offset: int) -> int:
    """Where logical element `index` lives in storage: offset + sum_k index[k] * stride[k]."""
    return offset + sum(i * s for i, s in zip(index, stride))


def storage_indices(t: torch.Tensor) -> torch.Tensor:
    """For every logical element of `t`, the storage slot it reads.

    Returns a LongTensor with the same shape as `t`. Pure Python on purpose:
    it is the definition, not a fast path. Use it on small tensors.
    """
    out = torch.empty(t.shape, dtype=torch.long)
    for index in itertools.product(*(range(n) for n in t.shape)):
        out[index] = storage_index(index, t.stride(), t.storage_offset())
    return out


def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True when `a` and `b` read the same block of memory."""
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def transpose_manual(t: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Transpose by swapping two entries of shape and stride. No data moves."""
    shape = list(t.shape)
    stride = list(t.stride())
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    return t.as_strided(shape, stride, t.storage_offset())


def permute_manual(t: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Permute by reordering shape and stride with the same index list."""
    shape = [t.shape[d] for d in dims]
    stride = [t.stride(d) for d in dims]
    return t.as_strided(shape, stride, t.storage_offset())


def view_manual(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """A `view` for contiguous tensors: same storage, new row-major strides.

    `torch.Tensor.view` handles more cases (it can split or merge dims whose
    strides line up). This version only covers the common one, which is
    exactly the case where a view is always legal.
    """
    if not t.is_contiguous():
        raise RuntimeError("view_manual needs a contiguous tensor; call .contiguous() first")
    if torch.Size(shape).numel() != t.numel():
        raise RuntimeError(f"shape {shape} is invalid for input of size {t.numel()}")
    return t.as_strided(shape, contiguous_strides(shape), t.storage_offset())
