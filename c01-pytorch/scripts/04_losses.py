"""Cross entropy step by step, checked against F.cross_entropy.

Run: python c01-pytorch/scripts/04_losses.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from scratchnet.losses import cross_entropy, log_softmax, log_softmax_naive, mse_loss

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, -1.0], [1.2, 0.2, 3.0], [0.0, 0.0, 0.0]])
target = torch.tensor([0, 1, 1, 2])

print("logits [N=4, C=3]:\n", logits)
m = logits.amax(-1, keepdim=True)
print("row max m [4, 1]:", m.squeeze(1))
lse = m + (logits - m).exp().sum(-1, keepdim=True).log()
print("logsumexp [4, 1]:", lse.squeeze(1))
logp = log_softmax(logits)
print("log_softmax [4, 3]:\n", logp)
picked = logp.gather(1, target[:, None])
print("gather(1, target[:, None]) [4, 1]:", picked.squeeze(1))
print("ours  :", cross_entropy(logits, target).item())
print("torch :", F.cross_entropy(logits, target).item())

print("\n== ignore_index, for padded positions ==")
target_pad = torch.tensor([0, 1, -100, 2])
print("ours  :", cross_entropy(logits, target_pad).item())
print("torch :", F.cross_entropy(logits, target_pad).item())

print("\n== large logits: naive vs stable ==")
big = torch.tensor([[100.0, 90.0, 0.0]])
print("naive :", log_softmax_naive(big))
print("stable:", log_softmax(big))
print("torch :", F.log_softmax(big, -1))

print("\n== MSE ==")
pred, y = torch.randn(8, 3), torch.randn(8, 3)
print("ours  :", mse_loss(pred, y).item(), " torch:", F.mse_loss(pred, y).item())
