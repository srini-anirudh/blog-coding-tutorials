"""The LoRA layer itself (post: "LoRA", "Why B = 0 and A random", "The backward pass, written out", Fig. 3).

1. Four ways to initialise B A. Which ones start at the base model, and which ones can learn.
2. The three backward formulas, checked against autograd.
3. On the first step only B moves.
4. Which activations autograd keeps: a frozen linear layer does not need its input, an adapted one does.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loralab.lora import INITS, LoRALinear, manual_grads
from loralab.report import Report, table

D, H, N = 64, 128, 512


def make_task(seed=0):
    """A frozen 'pretrained' 2-layer MLP, and a teacher that differs by a rank-4 change to the first layer."""
    g = torch.Generator().manual_seed(seed)
    base = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, D))
    with torch.no_grad():
        for p in base.parameters():
            p.copy_(torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5)
    delta = (torch.randn(H, 4, generator=g) @ torch.randn(4, D, generator=g)) / D ** 0.5
    x = torch.randn(N, D, generator=g)
    with torch.no_grad():
        y = base[2](torch.nn.functional.gelu(x @ (base[0].weight + delta).T + base[0].bias))
    return base, x, y


def adapted(base, init, r=8, seed=1):
    torch.manual_seed(seed)
    model = nn.Sequential(LoRALinear(base[0], r=r, alpha=2 * r, init=init), base[1], base[2])
    for p in model[2].parameters():
        p.requires_grad_(False)
    return model


def saved_activation_bytes(layer: nn.Module, x: torch.Tensor) -> int:
    """Bytes autograd stashes for backward, not counting tensors that are the layer's own parameters."""
    params = {p.data_ptr() for p in layer.parameters()}
    total = 0

    def pack(t):
        nonlocal total
        if t.data_ptr() not in params:
            total += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        layer(x).sum()
    return total


def main():
    rep = Report("03_lora_layer")
    base, x, y = make_task()
    with torch.no_grad():
        y_base = base(x)
    rep("## Part 1: four initialisations (64 -> 128 -> 64 MLP, adapter r=8 on the first layer, Adam lr 1e-2, 300 steps)")
    rows = []
    for init in INITS:
        model = adapted(base, init)
        lora = model[0]
        with torch.no_grad():
            drift = (model(x) - y_base).abs().max().item()
        loss0 = nn.functional.mse_loss(model(x), y)
        loss0.backward()
        gA, gB = lora.lora_A.grad.norm().item(), lora.lora_B.grad.norm().item()
        opt = torch.optim.Adam([lora.lora_A, lora.lora_B], lr=1e-2)
        curve = {}
        for step in range(1, 301):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            curve[step] = loss.item()
        rows.append([init, drift, loss0.item(), gA, gB, curve[10], curve[50], curve[300]])
    for line in table(rows, ["init", "max |h - h_base| at step 0", "loss at step 0", "||dL/dA||", "||dL/dB||",
                             "loss step 10", "loss step 50", "loss step 300"]):
        rep(line)
    rep("(the loss at step k is measured before the k-th update)")
    rep.data["inits"] = rows

    rep("\n## Part 2: the backward formulas vs autograd (float64, n=32, 48 -> 40, r=6, s=alpha/r=2)")
    torch.manual_seed(0)
    W, A, B = torch.randn(40, 48, dtype=torch.float64), torch.randn(6, 48, dtype=torch.float64), \
        torch.randn(40, 6, dtype=torch.float64)
    xx = torch.randn(32, 48, dtype=torch.float64, requires_grad=True)
    A.requires_grad_(), B.requires_grad_()
    s = 2.0
    h = xx @ W.T + s * (xx @ A.T) @ B.T
    g = torch.randn_like(h)
    h.backward(g)
    dB, dA, dx = manual_grads(W, A.detach(), B.detach(), s, xx.detach(), g)
    diffs = {"dL/dB": (dB - B.grad).abs().max().item(), "dL/dA": (dA - A.grad).abs().max().item(),
             "dL/dx": (dx - xx.grad).abs().max().item()}
    for k, v in diffs.items():
        rep(f"{k}: max |manual - autograd| = {v:.2e}")
    rep.data["backward_check"] = diffs

    rep("\n## Part 3: one Adam step from B = 0")
    model = adapted(base, "b_zero")
    lora = model[0]
    A0, B0 = lora.lora_A.detach().clone(), lora.lora_B.detach().clone()
    opt = torch.optim.Adam([lora.lora_A, lora.lora_B], lr=1e-2)
    nn.functional.mse_loss(model(x), y).backward()
    opt.step()
    moved = {"A": (lora.lora_A - A0).abs().max().item(), "B": (lora.lora_B - B0).abs().max().item()}
    rep(f"after step 1: max change in A = {moved['A']:.3g}, in B = {moved['B']:.3g}")
    opt.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    opt.step()
    rep(f"after step 2: max change in A since init = {(lora.lora_A - A0).abs().max().item():.3g}")
    rep.data["first_step"] = moved

    rep("\n## Part 4: activations kept for backward by one 1024 -> 1024 layer, input (256, 1024) fp32 = 1.05 MB")
    xin = torch.randn(256, 1024, requires_grad=True)  # the input needs a gradient, as inside a deep network
    rows = []
    for label, layer in (("trainable nn.Linear", nn.Linear(1024, 1024)),
                         ("frozen nn.Linear", nn.Linear(1024, 1024).requires_grad_(False)),
                         ("frozen + LoRA r=8", LoRALinear(nn.Linear(1024, 1024), r=8)),
                         ("frozen + LoRA r=64", LoRALinear(nn.Linear(1024, 1024), r=64))):
        rows.append([label, saved_activation_bytes(layer, xin) / 1e6])
    for line in table(rows, ["layer", "saved activations MB"]):
        rep(line)
    rep.data["saved_activations_mb"] = rows
    rep.save()


if __name__ == "__main__":
    main()
