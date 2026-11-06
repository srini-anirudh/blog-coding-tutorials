"""Merging, swapping, serving (post: "Merging, swapping, serving", "Unmerging and its hazards", Fig. 6).

1. Merging is exact: W + s B A gives the same outputs as the unmerged branch, up to rounding.
2. Hot-swapping in bf16 drifts: merge adapter 1, unmerge it, merge adapter 2, unmerge it, many times.
   The recovered base is not the base you started with. In fp32 the drift is far smaller.
3. Merging into an NF4 base: dequantize, add, re-quantize. How much of the update survives?
4. Many adapters, one base: one shared GEMM plus a gathered low-rank branch per request equals running each request
   through its own merged model.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loralab.lora import LoRALinear, multi_adapter_forward
from loralab.quant import NF4_PUBLISHED, dequantize_blockwise, quantize_blockwise
from loralab.report import Report, table

D, R = 1024, 16


def random_adapter(seed, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(R, D, generator=g) * scale, torch.randn(D, R, generator=g) * scale


def main():
    rep = Report("05_merge_serve")
    torch.manual_seed(0)
    base = nn.Linear(D, D)
    with torch.no_grad():
        base.weight.normal_(0, D ** -0.5)

    rep(f"## Part 1: merge exactness, {D} x {D} layer, r={R}, alpha={2 * R}, fp32")
    layer = LoRALinear(base, r=R, alpha=2 * R)
    A, B = random_adapter(1)
    layer.lora_A.data.copy_(A)
    layer.lora_B.data.copy_(B)
    x = torch.randn(64, D)
    with torch.no_grad():
        y_branch = layer(x)
        layer.merge()
        y_merged = layer(x)
        layer.unmerge()
    diff = (y_branch - y_merged).abs().max().item()
    rep(f"max |unmerged - merged| output = {diff:.2e} (output scale {y_branch.abs().max().item():.2f})")
    rep.data["merge_exact_max_diff"] = diff

    rep("\n## Part 2: hot-swap drift. Each cycle: merge adapter 1, unmerge, merge adapter 2, unmerge")
    W0 = base.weight.detach().clone()
    deltas = [layer.scale * (B_ @ A_) for A_, B_ in (random_adapter(1), random_adapter(2))]
    rows = []
    for dtype in (torch.bfloat16, torch.float32):
        W = W0.to(dtype).clone()
        start = W.clone()
        ds = [d.to(dtype) for d in deltas]
        for cycle in range(1, 1001):
            for d in ds:
                W += d
                W -= d
            if cycle in (1, 10, 100, 1000):
                err = (W.float() - start.float())
                rows.append([str(dtype).split(".")[1], cycle, err.abs().max().item(),
                             (err.norm() / start.float().norm()).item(), (err != 0).float().mean().item()])
    for line in table(rows, ["dtype", "cycles", "max |W - W_start|", "relative Frobenius drift", "share of entries changed"]):
        rep(line)
    rep(f"for scale: ||s B A||_F / ||W||_F = {(deltas[0].norm() / W0.norm()).item():.4f}")
    rep.data["hot_swap_drift"] = rows

    rep("\n## Part 3: merging into an NF4 base (blocks of 64): dequantize, add s B A, re-quantize")
    codes, absmax = quantize_blockwise(W0, NF4_PUBLISHED, 64)
    Wq = dequantize_blockwise(codes, absmax, NF4_PUBLISHED, W0.shape)
    rep(f"NF4 error of the base itself: ||Q(W) - W||_F / ||W||_F = {((Wq - W0).norm() / W0.norm()).item():.4f}")
    rows = []
    for rel in (0.001, 0.01, 0.05, 0.2):
        d = deltas[0] * (rel * W0.norm() / deltas[0].norm())
        c2, a2 = quantize_blockwise(Wq + d, NF4_PUBLISHED, 64)
        Wm = dequantize_blockwise(c2, a2, NF4_PUBLISHED, W0.shape)
        landed = Wm - Wq                                          # what the merge actually added
        kept = ((landed * d).sum() / d.pow(2).sum()).item()       # projection of the landed change on the intended one
        noise = (landed - d).norm().item() / d.norm().item()
        rows.append([rel, kept, noise, (landed != 0).float().mean().item()])
    for line in table(rows, ["||dW|| / ||W||", "share of update kept", "||error|| / ||dW||", "entries that changed"]):
        rep(line)
    rep("'share kept' is <landed, dW> / ||dW||^2. 'error' is the re-quantization error measured against dW itself.")
    rep.data["nf4_merge"] = rows

    rep("\n## Part 4: one base, 8 adapters, a mixed batch of 32 requests")
    K = 8
    As = torch.stack([random_adapter(10 + k)[0] for k in range(K)])
    Bs = torch.stack([random_adapter(10 + k)[1] for k in range(K)])
    ids = torch.randint(0, K, (32,), generator=torch.Generator().manual_seed(3))
    xb = torch.randn(32, D)
    s = layer.scale
    with torch.no_grad():
        y_batched = multi_adapter_forward(xb, W0, As, Bs, ids, s)
        y_ref = torch.stack([xb[i] @ (W0 + s * Bs[ids[i]] @ As[ids[i]]).T for i in range(32)])
    diff = (y_batched - y_ref).abs().max().item()
    stored = K * 2 * R * D
    rep(f"max |batched - per-request merged| = {diff:.2e}")
    rep(f"memory: one base ({D * D / 1e6:.2f} M) + {K} adapters ({stored / 1e6:.2f} M) vs {K} merged copies "
        f"({K * D * D / 1e6:.2f} M)")
    rep.data["multi_adapter_max_diff"] = diff
    rep.save()


if __name__ == "__main__":
    main()
