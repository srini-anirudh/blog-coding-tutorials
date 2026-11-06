"""What autograd keeps, and the API switches that change it.

1. Bytes each op saves for its backward pass, measured with saved-tensor hooks.
2. Attention's saved bill as the sequence grows, with P saved and with P recomputed.
3. Freezing early against late layers, detach, no_grad and inference_mode: nodes recorded and bytes saved.
4. Intermediate gradients are freed; .grad on leaves accumulates.
5. The version counter: which in-place writes are caught and which are safe.
6. Gradient accumulation: exact with 1/K, K times too large without it, and not exact with BatchNorm.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from autogradlab.report import Report  # noqa: E402
from autogradlab.saved import graph_nodes, saved_bytes  # noqa: E402
from autogradlab.vjp import AttentionRecompute, attention_forward  # noqa: E402

rep = Report("06_what_autograd_keeps")
torch.manual_seed(0)
B, n = 64, 256
act = B * n * 4  # bytes in one float32 activation of shape (B, n)
W = torch.randn(n, n, requires_grad=True)
g, b = torch.ones(n, requires_grad=True), torch.zeros(n, requires_grad=True)


def x():
    return torch.randn(B, n, requires_grad=True)


ops = [
    ("u = v W", lambda v: v @ W),
    ("u = v1 + v2", lambda v: v + v.detach().clone().requires_grad_(True)),
    ("u = v1 * v2", lambda v: v * v.detach().clone().requires_grad_(True)),
    ("ReLU", F.relu),
    ("GELU", F.gelu),
    ("LayerNorm", lambda v: F.layer_norm(v, (n,), g, b)),
    ("Softmax", lambda v: torch.softmax(v, -1)),
]
rows = []
for name, fn in ops:
    nbytes, saves = saved_bytes(fn, x(), exclude=[W, g, b])
    rows.append([name, saves, nbytes, f"{nbytes / act:.2f}"])
rep(f"1. Saved for backward, per op, input shape ({B}, {n}) float32. Parameters are not counted.")
rep.table(["op", "tensors saved", "bytes", "activations"], rows)
rep("Addition saves nothing. ReLU saves its output (a full float tensor, not one bit). LayerNorm saves its input")
rep("plus a mean and a reciprocal std per row, which is the 0.02 over 1.")

rep("\n2. Attention, 4 heads of dim 64, float32, batch 1. Bytes saved, as sequence length N grows:")
rows = []
for N in [128, 256, 512, 1024]:
    qkv = [torch.randn(1, 4, N, 64, requires_grad=True) for _ in range(3)]
    plain, _ = saved_bytes(lambda q, k, v: attention_forward(q, k, v)[0], *qkv)
    recomp, _ = saved_bytes(AttentionRecompute.apply, *qkv)
    p_bytes = 4 * N * N * 4
    rows.append([N, f"{plain / 2**20:.2f}", f"{recomp / 2**20:.2f}", f"{p_bytes / 2**20:.2f}"])
rep.table(["N", "P saved (MiB)", "P recomputed (MiB)", "one P (MiB)"], rows)
rep("With P saved, the bill is Q, K, V plus one P, and P dominates by N = 1024. With P recomputed it is Q, K and V")
rep("only, so it grows 2x per doubling of N instead of nearly 4x.")


class Net(nn.Module):
    def __init__(self, depth=6, width=n):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(width, width), nn.GELU()) for _ in range(depth))
        self.head = nn.Linear(width, 10)

    def forward(self, h, detach_before_head=False, no_grad_block=None):
        for i, blk in enumerate(self.blocks):
            if i == no_grad_block:
                with torch.no_grad():
                    h = blk(h)
            else:
                h = blk(h)
        if detach_before_head:
            h = h.detach()
        return self.head(h)


net = Net()
params = list(net.parameters())


def measure(setup=lambda: None, **kw):
    for p in params:
        p.requires_grad_(True)
    setup()
    out = {}
    nbytes, _ = saved_bytes(lambda inp: (out.setdefault("y", net(inp, **kw))), torch.randn(B, n), exclude=params)
    return graph_nodes(out["y"]), nbytes


rows = [["nothing frozen", *measure()]]
for k in [1, 3, 5]:
    rows.append([f"first {k} of 6 blocks frozen",
                 *measure(lambda k=k: [p.requires_grad_(False) for blk in net.blocks[:k] for p in blk.parameters()])])
rows.append(["last 3 of 6 blocks frozen",
             *measure(lambda: [p.requires_grad_(False) for blk in net.blocks[3:] for p in blk.parameters()])])
rows.append(["detach() before the head", *measure(detach_before_head=True)])
rows.append(["block 2 under no_grad()", *measure(no_grad_block=2)])
rep("\n3. Six Linear+GELU blocks and a head. Nodes recorded and bytes saved for one forward pass:")
rep.table(["setting", "graph nodes", "saved bytes"], [[r[0], r[1], r[2]] for r in rows])
rep("Freezing early blocks prunes the graph up to the first block that still needs a gradient: the first 3 frozen")
rep("cuts saved bytes by 46%. Freezing the last 3 cuts only 23%. Those blocks must still pass the gradient back, so")
rep("they keep W and the GELU input. Only the Linear input, needed for dW alone, is dropped.")
rep("detach() cuts the graph at one tensor, so only the head is recorded. no_grad() records nothing inside block 2,")
rep("which also cuts blocks 0 and 1 off from the loss.")
for p in params:
    p.requires_grad_(True)

with torch.inference_mode():
    t = torch.randn(3)
try:
    (t * torch.ones(3, requires_grad=True)).sum().backward()
    rep("inference_mode tensor used in autograd: allowed")
except RuntimeError as e:
    rep("inference_mode tensor used later with autograd: RuntimeError:", str(e).split(".")[0])

rep("\n4. Intermediate gradients are computed, used and freed. Leaf .grad accumulates:")
w = torch.tensor(2.0, requires_grad=True)
h = w * 3
(h**2).backward()
rep(f"  after one backward:  w.grad = {w.grad.item()}, h.grad = {h.grad}")
h = w * 3
h.retain_grad()
(h**2).backward()
rep(f"  after a second one, no zero_grad, h.retain_grad(): w.grad = {w.grad.item()} (= 36 + 36), h.grad = {h.grad.item()}")

rep("\n5. The version counter:")
cases = {
    "sigmoid output modified in place": lambda v: (lambda y: (y.mul_(2), y)[1])(torch.sigmoid(v)),
    "x modified after y = x ** 2": lambda v: (lambda a: (lambda y: (a.add_(1), y)[1])(a**2))(v * 1),
    "relu_() on a linear output": lambda v: (v @ W).relu_(),
    "add_(1) on a tensor nobody saved": lambda v: (v + 1).add_(1) * 2,
}
for name, fn in cases.items():
    try:
        fn(x()).sum().backward()
        rep(f"  {name:<34} ok, gradient computed")
    except RuntimeError as e:
        msg = str(e).split("\n")[0]
        rep(f"  {name:<34} RuntimeError: ...{msg[msg.find('is at version'):].split(' instead')[0]}")

rep("\n6. Gradient accumulation over K = 4 microbatches of 16, against one batch of 64:")
data, target = torch.randn(64, 32), torch.randn(64, 1)
rows = []
for norm in ["LayerNorm", "BatchNorm1d"]:
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(32, 32), getattr(nn, norm)(32), nn.Tanh(), nn.Linear(32, 1))
    model.train()

    def flat_grad():
        return torch.cat([p.grad.reshape(-1) for p in model.parameters()])

    model.zero_grad()
    F.mse_loss(model(data), target).backward()
    full = flat_grad().clone()
    for scale in [True, False]:
        model.zero_grad()
        for xs, ys in zip(data.split(16), target.split(16)):
            loss = F.mse_loss(model(xs), ys)
            (loss / 4 if scale else loss).backward()
        acc = flat_grad()
        rows.append([norm, "loss / K" if scale else "no 1/K", f"{(acc - full).norm() / full.norm():.1e}",
                     f"{acc.norm() / full.norm():.3f}"])
rep.table(["norm layer", "scaling", "relative error", "norm ratio"], rows)
rep("With LayerNorm and 1/K the accumulated gradient matches the full batch to float32 rounding. Without 1/K it is")
rep("K = 4 times too large. BatchNorm computes statistics over whatever is in the microbatch, so it is not exact.")
rep.save()
