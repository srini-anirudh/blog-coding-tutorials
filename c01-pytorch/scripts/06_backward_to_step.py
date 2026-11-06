"""Trace one training step from zero_grad to step, printing every piece of state.

Run: python c01-pytorch/scripts/06_backward_to_step.py [out.json]
The optional file gets head.weight before and after the step, for plotting.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scratchnet.autograd_tools import walk_graph
from scratchnet.data import SpiralDataset
from scratchnet.debug import ActivationStats, GradNormMonitor
from scratchnet.layers import MLP, ReLU
from scratchnet.losses import cross_entropy
from scratchnet.optim import grad_norm

torch.manual_seed(0)
ds = SpiralDataset(100)
x, y = ds.x[:8], ds.y[:8]  # one batch: x [8, 2], y [8]
model = MLP(2, 16, 3, depth=2, mean=ds.x.mean(0), std=ds.x.std(0))
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
w = model.head.weight  # follow one parameter through the step: [16, 3]

print("== 0. before anything ==")
print("head.weight.grad:", w.grad, "| optimizer state entries:", len(opt.state))

print("\n== 1. forward ==")
with ActivationStats(model, (ReLU,)) as acts:
    logits = model(x)
loss = cross_entropy(logits, y)
nodes = walk_graph(loss)
print(f"logits {list(logits.shape)}  loss {loss.item():.4f}  graph nodes {len(nodes)}  "
      f"leaves {sum(n.leaf_shape is not None for n in nodes)}")
for name, st in acts.stats.items():
    print(f"  {name} (ReLU) output: mean {st.mean:.3f}  std {st.std:.3f}  zeros {st.frac_zero:.0%}")
print("still no gradients:", all(p.grad is None for p in model.parameters()))

print("\n== 2. loss.backward() ==")
with GradNormMonitor(model) as mon:
    loss.backward()
print("order in which .grad was written:")
for i, name in enumerate(mon.record.order):
    p = dict(model.named_parameters())[name]
    print(f"  {i}: {name:<18} grad {str(list(p.grad.shape)):<8} norm {mon.record.norms[name].item():.4f}")
try:
    loss.backward()
except RuntimeError:
    print("a second loss.backward() fails: the graph's saved tensors were freed")

print("\n== 3. between backward and step ==")
print(f"global grad norm {grad_norm(model.parameters()).item():.4f}  (clip, unscale or all-reduce would happen here)")
print("optimizer state entries:", len(opt.state), "(state is created lazily on the first step)")
w_before = w.detach().clone()
g = w.grad.detach().clone()

print("\n== 4. optimizer.step() ==")
opt.step()
state = opt.state[w]
print("state keys for head.weight:", {k: (list(v.shape) if v.dim() else v.item()) for k, v in state.items()})
state_bytes = sum(v.numel() * v.element_size() for s in opt.state.values() for v in s.values() if v.dim())
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"param bytes {param_bytes}  optimizer state bytes {state_bytes}  ratio {state_bytes / param_bytes:.1f}")

# Re-derive one element of the update from the AdamW formula: the one with the largest gradient.
lr, wd, (b1, b2), eps = 1e-2, 0.01, (0.9, 0.999), 1e-8
i, j = divmod(g.abs().argmax().item(), g.shape[1])
p0, g0 = w_before[i, j].item(), g[i, j].item()
m = (1 - b1) * g0
v = (1 - b2) * g0 * g0
p_hand = p0 * (1 - lr * wd) - lr * (m / (1 - b1)) / ((v / (1 - b2)) ** 0.5 + eps)
print(f"head.weight[{i}, {j}]: before {p0:.6f}  grad {g0:.6f}  after {w[i, j].item():.6f}  by hand {p_hand:.6f}")
delta = (w.detach() - w_before).abs()
moved = g != 0
print(f"|update| where grad != 0: mean {delta[moved].mean():.5f} (first step ~ lr = 0.01)  "
      f"| entries with grad == 0: {(~moved).sum().item()} of {g.numel()}, moved by {delta[~moved].max():.1e}")

print("\n== 5. optimizer.zero_grad() ==")
opt.zero_grad()
print("head.weight.grad:", w.grad, "| state survives:", int(opt.state[w]["step"].item()), "step(s) recorded")

if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps({
        "before": w_before.tolist(), "grad": g.tolist(), "after": w.tolist(),
        "m": state["exp_avg"].tolist(), "v": state["exp_avg_sq"].tolist(),
    }))
