"""Autograd, inspected: leaves, grad_fn, the graph, accumulation, retain_graph, detach.

Run: python c01-pytorch/scripts/03_autograd.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scratchnet.autograd_tools import format_graph
from scratchnet.layers import Linear

torch.manual_seed(0)

print("== a three-op graph ==")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
y = x * w  # MulBackward0
z = y.sum()  # SumBackward0
for name, t in [("x", x), ("w", w), ("y", y), ("z", z)]:
    print(f"{name}: is_leaf={t.is_leaf!s:<5} requires_grad={t.requires_grad!s:<5} grad_fn={type(t.grad_fn).__name__}")
print("z.grad_fn.next_functions:", [type(fn).__name__ for fn, _ in z.grad_fn.next_functions])
print("y.grad_fn.next_functions:", [type(fn).__name__ for fn, _ in y.grad_fn.next_functions])
print(format_graph(z))

y.retain_grad()  # non-leaf grads are freed by default; ask to keep this one
z.backward()
print("x.grad =", x.grad.tolist(), " (dz/dx = w)")
print("w.grad =", w.grad.tolist(), " (dz/dw = x)")
print("y.grad =", y.grad.tolist())

print("\n== the graph of a Linear layer and a loss ==")
layer = Linear(3, 2)
inp = torch.randn(4, 3)
loss = layer(inp).pow(2).mean()
print(format_graph(loss))

print("\n== gradients accumulate ==")
w = torch.tensor(2.0, requires_grad=True)
for i in range(3):
    (3 * w).backward()
    print(f"after backward #{i + 1}: w.grad = {w.grad.item()}")
w.grad = None
print("after w.grad = None:", w.grad)

print("\n== backward frees the graph ==")
a = torch.tensor(1.5, requires_grad=True)
b = a.exp()
b.backward()
try:
    b.backward()
except RuntimeError as err:
    print("second backward ->", str(err).split(".")[0])
a.grad = None
b = a.exp()
b.backward(retain_graph=True)
b.backward()
print(f"with retain_graph=True: a.grad = {a.grad.item():.4f} = 2 * e^1.5")

print("\n== detach and no_grad ==")
x = torch.randn(3, requires_grad=True)
print("x.detach().requires_grad:", x.detach().requires_grad)
with torch.no_grad():
    y = x * 2
print("under no_grad: y.requires_grad =", y.requires_grad, " y.grad_fn =", y.grad_fn)
