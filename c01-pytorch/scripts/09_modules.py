"""What nn.Module tracks: parameters, submodules, buffers, state_dict, train and eval.

Run: python c01-pytorch/scripts/09_modules.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scratchnet.layers import MLP, Dropout, count_parameters

torch.manual_seed(0)

model = MLP(2, 16, 3)
print(model)

print("\n== submodules ==")
print([name for name, _ in model.named_children()])

print("\n== parameters ==")
for name, p in model.named_parameters():
    print(f"{name:<18} {str(list(p.shape)):<8} {p.numel():>4}")
print("count_parameters:", count_parameters(model))

print("\n== buffers: saved and moved, but never optimized ==")
for name, buf in model.named_buffers():
    print(f"{name:<18} {buf.tolist()}")
param_ids = {id(p) for p in model.parameters()}
print("norm.mean returned by .parameters():", id(model.norm.mean) in param_ids)

print("\n== state_dict: parameters and buffers together ==")
print(list(model.state_dict().keys()))

print("\n== train() and eval() set one flag on every submodule ==")
print("new model: model.training =", model.training, " head.training =", model.head.training)
model.eval()
print("after eval(): model.training =", model.training, " head.training =", model.head.training)
model.train()
print("after train(): model.training =", model.training, " head.training =", model.head.training)

print("\n== the flag only matters to layers that read it: Dropout(0.5) ==")
drop = Dropout(0.5)
x = torch.ones(8)
print("train mode:", drop(x).tolist())
drop.eval()
print("eval mode: ", drop(x).tolist())
