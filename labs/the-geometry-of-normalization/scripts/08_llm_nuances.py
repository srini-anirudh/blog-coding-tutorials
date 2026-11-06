"""Section 11: details that bite in a real training run, each checked numerically.

  1. Where eps sits: x / sqrt(ms + eps) vs x / (sqrt(ms) + eps).
  2. Reduce in fp32: RMSNorm of 4096-dim vectors with values around 100, in bf16 with and without upcasting.
  3. Gemma's (1 + w) gain: loading it into a plain-gain RMSNorm, and what weight decay pulls each one towards.
  4. Weight-decay groups: which parameters of a model get decay.
  5. Folding: RMSNorm's gain merges into the next bias-free linear layer, and the division can wait until after it.
  6. With QK-Norm, the pre-attention RMSNorm is redundant on the q and k paths (but not on v).
"""

import _setup  # noqa: F401
import torch
import torch.nn.functional as F

from normgeo.blocks import TinyLM
from normgeo.ops import RMSNorm, weight_decay_groups
from normgeo.report import Report

torch.manual_seed(0)
D = 4096

with Report("08_llm_nuances") as say:
    say("1. eps placement, eps = 1e-6. Max relative difference between the two conventions.")
    inside, outside = RMSNorm(256, eps=1e-6), RMSNorm(256, eps=1e-6, eps_inside=False)
    rows = []
    for r in [1e-4, 1e-3, 1e-2, 1.0, 100.0]:
        x = torch.randn(64, 256, dtype=torch.float64) * r
        with torch.no_grad():
            a, b = inside(x), outside(x)
        rel = ((a - b).abs() / b.abs().clamp_min(1e-30)).max().item()
        rows.append(rel)
        say(f"   input RMS {r:>7g}:  {rel:.1e}")
    say("   They agree for normal activations and disagree badly when the RMS is near sqrt(eps).")

    say()
    say(f"2. RMSNorm in bf16, {D}-dim vectors, values ~ N(0, 100^2). Error vs a float64 reference.")
    x64 = torch.randn(256, D, dtype=torch.float64) * 100
    ref = x64 / x64.pow(2).mean(-1, keepdim=True).sqrt()
    xb = x64.to(torch.bfloat16)
    with torch.no_grad():
        up, no_up = RMSNorm(D, eps=0)(xb).double(), RMSNorm(D, eps=0, upcast=False)(xb).double()
    say(f"   upcast to fp32 inside:   max rel error {((up - ref).abs() / ref.abs()).max():.2e}   (bf16 output rounding only)")
    say(f"   reduce in bf16 (torch):  max rel error {((no_up - ref).abs() / ref.abs()).max():.2e}")
    # torch's CPU mean may accumulate in fp32 internally. A plain bf16 running sum shows the worst case.
    v = xb[0].pow(2)
    acc = torch.zeros((), dtype=torch.bfloat16)
    for t in v:
        acc = acc + t
    true_ms = x64[0].pow(2).mean().item()
    say(f"   mean(x^2) by a bf16 running sum: {acc.item() / D:.4g} vs true {true_ms:.4g} "
        f"({abs(acc.item() / D - true_ms) / true_ms:.0%} off). Once the sum is large, small terms round to nothing.")

    say()
    say("3. Gemma's (1 + w) gain, with trained-looking w ~ N(0, 0.05^2)")
    gemma, llama = RMSNorm(512, offset=True), RMSNorm(512)
    with torch.no_grad():
        gemma.weight.copy_(0.05 * torch.randn(512))
        llama.weight.copy_(gemma.weight)  # the porting bug: copy the stored tensor as if it were the gain
        x = torch.randn(32, 512)
        good, bad = gemma(x), llama(x)
    say(f"   output RMS: Gemma layer {good.pow(2).mean().sqrt():.3f}, same tensor in a plain-gain layer {bad.pow(2).mean().sqrt():.3f}")
    say(f"   relative error of the ported layer: {(bad - good).norm() / good.norm():.3f}")
    for name, layer in [("Gemma (1 + w)", RMSNorm(8, offset=True)), ("plain w", RMSNorm(8))]:
        with torch.no_grad():
            layer.weight.fill_(0.5 if name == "plain w" else -0.5)  # both start at gain 0.5
        opt = torch.optim.AdamW(layer.parameters(), lr=1e-2, weight_decay=0.1)
        for _ in range(2000):  # no loss: only decay acts
            layer.weight.grad = torch.zeros_like(layer.weight)
            opt.step()
        gain = (1 + layer.weight if layer.offset else layer.weight).mean().item()
        say(f"   decay only, 2000 AdamW steps (lr 1e-2, wd 0.1), gain 0.5 -> {gain:.3f}   [{name}]")
    say("   Decay pulls the plain gain towards 0 and the Gemma gain towards 1.")

    say()
    model = TinyLM(65, 64, 4, 4, "pre", 64)
    groups = weight_decay_groups(model, 0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    kinds = sorted({names[id(p)].split(".")[-2] if "blocks" in names[id(p)] else names[id(p)] for p in groups[1]["params"]})
    say("4. Weight-decay groups for a 4-block TinyLM")
    say(f"   decay:    {len(groups[0]['params'])} tensors, {sum(p.numel() for p in groups[0]['params']):,} parameters (all matrices)")
    say(f"   no decay: {len(groups[1]['params'])} tensors, {sum(p.numel() for p in groups[1]['params']):,} parameters: {', '.join(kinds)}")

    say()
    say("5. Folding an RMSNorm gain into the next bias-free linear layer, fp64")
    x = torch.randn(64, 512, dtype=torch.float64) * 3
    norm = RMSNorm(512, eps=0).double()
    with torch.no_grad():
        norm.weight.copy_(1 + 0.2 * torch.randn(512))
        W = torch.randn(1024, 512, dtype=torch.float64) / 512**0.5
        ref = norm(x) @ W.T
        W_fold = W * norm.weight  # W diag(gamma)
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        say(f"   W (g * x/rms)  vs  (W diag g)(x/rms):   {(ref - (x / rms) @ W_fold.T).abs().max():.1e}")
        say(f"   W (g * x/rms)  vs  ((W diag g) x)/rms:  {(ref - (x @ W_fold.T) / rms).abs().max():.1e}   (division after the GEMM)")

    say()
    say("6. With QK-Norm, the RMSNorm before attention is redundant on the q and k paths, fp64")
    with torch.no_grad():
        Wq, Wv = torch.randn(64, 512, dtype=torch.float64), torch.randn(64, 512, dtype=torch.float64)
        head_norm = lambda t: F.rms_norm(t, (64,))  # noqa: E731
        q_with = head_norm(norm(x) @ Wq.T)
        q_without = head_norm(x @ (Wq * norm.weight).T)  # pre-norm deleted, its gain folded into Wq
        v_with, v_without = norm(x) @ Wv.T, x @ (Wv * norm.weight).T
        say(f"   q path, pre-norm kept vs deleted + folded:  {(q_with - q_without).abs().max():.1e}")
        say(f"   v path, pre-norm kept vs deleted + folded:  {((v_with - v_without).norm() / v_with.norm()):.2f} relative")
    say("   The head norm divides out the scale the pre-norm removed, so on q and k it is redundant. v has no head norm.")
    say.json({"eps_rel_diff": rows})
