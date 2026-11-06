"""A linear layer from bare parameters, trained by vanilla SGD, then wrapped as a Module.

Run: python c01-pytorch/scripts/08_linear_manual.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from scratchnet.layers import Linear

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

print("== y = x W + b with bare parameters ==")
W = nn.Parameter(torch.randn(3, 2))  # [in, out]
b = nn.Parameter(torch.zeros(2))  # [out]
x = torch.randn(256, 3)  # [B, in]
y = x @ W + b
print("x", list(x.shape), "@ W", list(W.shape), "+ b", list(b.shape), "-> y", list(y.shape))

print("\n== vanilla SGD on a known target ==")
true_W = torch.tensor([[1.0, -2.0], [0.5, 0.0], [-1.0, 3.0]])
true_b = torch.tensor([0.5, -0.5])
target = x @ true_W + true_b
lr = 0.1
try:
    W -= lr * torch.ones_like(W)  # outside no_grad, autograd would have to record the update
except RuntimeError as err:
    print("without no_grad ->", err)
for step in range(201):
    loss = ((x @ W + b - target) ** 2).mean()
    loss.backward()
    with torch.no_grad():  # the update is not part of the model
        for p in (W, b):
            p -= lr * p.grad
            p.grad = None  # backward adds to .grad, so clear it before the next one
    if step % 50 == 0:
        print(f"step {step:>3}: loss {loss.item():.6f}")
print("recovered W:\n", W.detach())
print("true W:\n", true_W)
print("recovered b:", b.detach(), " true b:", true_b)
print(f"largest error: W {(W - true_W).abs().max().item():.2e}  b {(b - true_b).abs().max().item():.2e}")

print("\n== the same layer as a Module, checked against nn.Linear ==")
ours = Linear(3, 2)
ref = nn.Linear(3, 2)
print(ours)
print("ours weight", list(ours.weight.shape), " nn.Linear weight", list(ref.weight.shape), "(transposed)")
print("outputs equal before copying the weights:", torch.allclose(ours(x), ref(x)))
with torch.no_grad():
    ref.weight.copy_(ours.weight.t())
    ref.bias.copy_(ours.bias)
print("outputs equal after copying the transposed weight:", torch.allclose(ours(x), ref(x)))
