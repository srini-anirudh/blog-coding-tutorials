"""Broadcasting: the two shapes attention needs, and the bug it hides.

Run: python c01-pytorch/scripts/02_broadcasting.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scratchnet.broadcasting import broadcast_shape, expand_to

torch.manual_seed(0)
B, H, N, D = 2, 4, 5, 8

print("== [B, N, D] + [D]: a bias added to every token ==")
x = torch.randn(B, N, D)
bias = torch.randn(D)
print("broadcast_shape:", broadcast_shape(tuple(x.shape), tuple(bias.shape)))
e = expand_to(bias, (B, N, D))
print("bias expanded:", tuple(e.shape), "stride", e.stride())
torch.testing.assert_close(x + bias, x + e)

print("\n== [B, H, N, D] + [1, H, 1, D]: one vector per head ==")
q = torch.randn(B, H, N, D)
per_head = torch.randn(1, H, 1, D)
print("broadcast_shape:", broadcast_shape(tuple(q.shape), tuple(per_head.shape)))
e = expand_to(per_head, (B, H, N, D))
print("per_head expanded:", tuple(e.shape), "stride", e.stride())
out = q + per_head
added = per_head[0, 2, 0]  # the head-2 vector
same = torch.allclose(out[0, 2, 3], q[0, 2, 3] + added) and torch.allclose(out[1, 2, 0], q[1, 2, 0] + added)
print("batch 0 token 3 and batch 1 token 0 both get the head-2 vector:", same)

print("\n== a mismatch is an error ==")
try:
    broadcast_shape((B, N, D), (N,))
except ValueError as err:
    print("ValueError:", err)

print("\n== the silent bug: [B, 1] - [B] ==")
pred = torch.randn(4, 1)  # a regression head returns [B, 1]
target = torch.randn(4)  # labels come as [B]
diff = pred - target
print("pred", tuple(pred.shape), "- target", tuple(target.shape), "->", tuple(diff.shape))
print(f"loss on [B, B]: {(diff ** 2).mean():.4f}   correct loss: {((pred.squeeze(1) - target) ** 2).mean():.4f}")
