"""Look inside tensors: metadata, memory, strides and views.

Run: python c01-pytorch/scripts/01_tensors.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scratchnet.tensors import describe, shares_storage, storage_indices

torch.manual_seed(0)
torch.set_num_threads(2)


def show(name: str, t: torch.Tensor, base: torch.Tensor) -> None:
    lay = describe(t)
    print(f"{name:<22} shape={str(list(lay.shape)):<10} stride={str(lay.stride):<10} "
          f"offset={lay.offset:<2} contiguous={str(lay.contiguous):<5} shares={shares_storage(t, base)}")


print("== metadata ==")
x = torch.randn(4, 8)
lay = describe(x)
print(f"shape={tuple(x.shape)} rank={x.dim()} dtype={x.dtype} device={x.device}")
print(f"numel={x.numel()} element_size={lay.element_size} B -> {lay.nbytes} B")
for dtype in (torch.float32, torch.bfloat16, torch.int8):
    print(f"  as {str(dtype):<15} {describe(x.to(dtype)).nbytes:>4} B")

print("\n== views of one storage ==")
base = torch.arange(12)
m = base.view(3, 4)
show("base.view(3, 4)", m, base)
show("m.t()", m.t(), base)
show("m.t().contiguous()", m.t().contiguous(), base)
show("m[:, 1:3]", m[:, 1:3], base)
show("m[1]", m[1], base)
show("m[:, ::2]", m[:, ::2], base)
show("base.view(2, 3, 2)", base.view(2, 3, 2), base)
show("...permute(2, 0, 1)", base.view(2, 3, 2).permute(2, 0, 1), base)

print("\n== which storage slot each element reads ==")
print("m.t():")
print(storage_indices(m.t()))
print("m[:, 1:3]:")
print(storage_indices(m[:, 1:3]))

print("\n== view vs reshape ==")
t = m.t()
try:
    t.view(12)
except RuntimeError as err:
    print("t.view(12) ->", str(err).split(".")[0])
r = t.reshape(12)
print("t.reshape(12) shares storage with m:", shares_storage(r, m))
print("t.contiguous().view(12):", t.contiguous().view(12).tolist())

print("\n== a slice keeps the whole storage alive ==")
big = torch.zeros(1000, 1000)
row = big[0]
print(f"row.nbytes={describe(row).nbytes} B  storage={describe(row).storage_bytes / 1e6:.1f} MB")
print(f"row.clone(): storage={describe(row.clone()).storage_bytes} B")
