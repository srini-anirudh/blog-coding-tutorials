"""Matrix multiplication: the definition, then the shapes attention will use.

Run: python c01-pytorch/scripts/11_matmul.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scratchnet.matmul import bmm_loop, matmul_manual, mm_flops, mm_loops, mm_rows

torch.manual_seed(0)

print("== [M, K] @ [K, N] -> [M, N] ==")
a, b = torch.randn(4, 6), torch.randn(6, 3)
ref = torch.mm(a, b)
print("a", tuple(a.shape), "b", tuple(b.shape), "->", tuple(ref.shape), " flops:", mm_flops(4, 6, 3))
for name, out in [("mm_loops", mm_loops(a, b)), ("mm_rows", mm_rows(a, b))]:
    print(f"{name:<9} matches torch.mm: {torch.allclose(out, ref)}")

print("\n== [B, M, K] @ [B, K, N] -> [B, M, N]: one independent matmul per batch element ==")
ba, bb = torch.randn(2, 4, 6), torch.randn(2, 6, 3)
ref = torch.bmm(ba, bb)
print("ba", tuple(ba.shape), "bb", tuple(bb.shape), "->", tuple(ref.shape))
print("bmm_loop matches torch.bmm:", torch.allclose(bmm_loop(ba, bb), ref))
print("row 1 of the batch is ba[1] @ bb[1]:", torch.allclose(ref[1], ba[1] @ bb[1]))

print("\n== matmul broadcasts the leading dims; bmm does not ==")
for sa, sb in [((2, 4, 6), (6, 3)), ((2, 1, 4, 6), (5, 6, 3))]:
    x, y = torch.randn(sa), torch.randn(sb)
    out = matmul_manual(x, y)
    print(f"{sa} @ {sb} -> {tuple(out.shape)}  matches torch.matmul: {torch.allclose(out, torch.matmul(x, y))}")
try:
    torch.bmm(ba, bb[0])
except RuntimeError as err:
    print("torch.bmm(ba, bb[0]) ->", err)

print("\n== the attention shape: [B, H, N, d] @ [B, H, d, N] -> [B, H, N, N] ==")
B, H, N, d = 2, 4, 5, 8
q, k = torch.randn(B, H, N, d), torch.randn(B, H, N, d)
print("q @ k.transpose(-2, -1):", tuple((q @ k.transpose(-2, -1)).shape))
