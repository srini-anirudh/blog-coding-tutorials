"""Section 09: attention entropy collapse, QK-Norm, and why QK-Norm can sit before RoPE.

Part A (Fig 08). Scale q and k by s, as a growing residual stream would. Logits grow as s^2, the softmax saturates,
and once it has saturated, the gradient that reaches the logits shrinks. With QK-Norm (RMSNorm over the head dim) nothing moves.
Part B. RoPE is a rotation, so it keeps the norm. RMSNorm(RoPE(q)) = RoPE(RMSNorm(q)) when the gain is 1.
A learned per-channel gain breaks the exact equality, because RoPE mixes channel pairs that have different gains.
For logit bounds and soft-capping, see c03-attention/scripts/stability_demo.py.
"""

import math

import _setup  # noqa: F401
import torch
import torch.nn.functional as F

from normgeo.geometry import rope
from normgeo.ops import RMSNorm
from normgeo.report import Report

torch.manual_seed(0)
H, T, D = 8, 256, 64
q0, k0, v = torch.randn(H, T, D), torch.randn(H, T, D), torch.randn(H, T, D)
G = torch.randn(H, T, D)  # a fixed upstream gradient on the attention output
qk_norm = RMSNorm(D)


def stats(q: torch.Tensor, k: torch.Tensor) -> dict:
    s = (q @ k.transpose(-1, -2) / math.sqrt(D)).requires_grad_(True)
    p = s.softmax(-1)
    ((p @ v) * G).sum().backward()
    ent = -(p * p.clamp_min(1e-30).log()).sum(-1)
    return {"max_logit": s.abs().max().item(), "entropy": ent.mean().item(), "max_prob": p.amax(-1).mean().item(),
            "grad": s.grad.norm().item()}


with Report("07_qk_norm") as say:
    say(f"Part A. {H} heads, {T} keys, head dim {D}. Uniform attention has entropy ln {T} = {math.log(T):.2f}.")
    say(f"{'scale s':>8} | {'no norm: max logit':>18}{'entropy':>9}{'max p':>8}{'||dL/dS||':>11} | "
        f"{'QK-Norm: max logit':>18}{'entropy':>9}{'max p':>8}{'||dL/dS||':>11}")
    rows = []
    for s in [1, 2, 4, 8, 16, 32]:
        a = stats(q0 * s, k0 * s)
        with torch.no_grad():
            qn, kn = qk_norm(q0 * s), qk_norm(k0 * s)
        b = stats(qn, kn)
        rows.append({"scale": s, "raw": a, "qk_norm": b})
        say(f"{s:>8} | {a['max_logit']:>18.1f}{a['entropy']:>9.2f}{a['max_prob']:>8.3f}{a['grad']:>11.3f} | "
            f"{b['max_logit']:>18.1f}{b['entropy']:>9.2f}{b['max_prob']:>8.3f}{b['grad']:>11.3f}")
    say("Without the norm, entropy collapses towards 0 and, past s = 2, the gradient on the logits falls as s grows.")
    say(f"With QK-Norm, |logit| <= ||q|| ||k|| / sqrt(d) = sqrt(d) = {math.sqrt(D):.0f} (gain 1). A learned gain can")
    say("still sharpen attention, but only by growing the gain itself, which is a parameter you can watch.")

    say()
    say("Part B. RoPE and RMSNorm, max abs difference over all heads and positions")
    q = q0 * 3
    say(f"  | ||RoPE(q)|| - ||q|| |                           {(rope(q).norm(dim=-1) - q.norm(dim=-1)).abs().max():.1e}")
    with torch.no_grad():
        say(f"  RMSNorm(RoPE(q)) vs RoPE(RMSNorm(q)), gain = 1     {(qk_norm(rope(q)) - rope(qk_norm(q))).abs().max():.1e}")
        g = RMSNorm(D)
        g.weight.copy_(1 + 0.3 * torch.randn(D))
        say(f"  same, gain ~ 1 + 0.3 N(0,1) per channel          {(g(rope(q)) - rope(g(q))).abs().max():.1e}")
        tied = RMSNorm(D)
        tied.weight.copy_((1 + 0.3 * torch.randn(D // 2)).repeat_interleave(2))
        say(f"  same, gain shared within each rotated pair       {(tied(rope(q)) - rope(tied(q))).abs().max():.1e}")
    say("  The scale commutes with the rotation. A per-channel gain does not, so the order is a real choice once")
    say("  the gain has trained. The post notes that Gemma 3 and OLMo 2 normalize first, then rotate.")
    say.json({"part_a": rows})
