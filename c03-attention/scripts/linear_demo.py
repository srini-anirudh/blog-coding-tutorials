"""Linear attention: same output by reordering matmuls, the recurrence, and random features."""

import math

import _setup  # noqa: F401
import torch

from attnzoo.linear import (
    causal_linear_attention_parallel,
    causal_linear_attention_recurrent,
    elu_feature_map,
    linear_attention,
    performer_features,
)

torch.manual_seed(0)
b, h, n, d, dv = 1, 2, 10, 8, 8
q, k, v = elu_feature_map(torch.randn(b, h, n, d)), elu_feature_map(torch.randn(b, h, n, d)), torch.randn(b, h, n, dv)

a = q @ k.transpose(-2, -1)
quadratic = (a @ v) / a.sum(-1, keepdim=True)
print(f"max |phi(Q)[phi(K)^T V] - [phi(Q)phi(K)^T]V| = {(linear_attention(q, k, v) - quadratic).abs().max().item():.1e}")
print("sizes: phi(K)^T V", tuple((k.transpose(-2, -1) @ v).shape), "against phi(Q) phi(K)^T", tuple(a.shape))

print("\nThe recurrence: state size stays fixed while the sequence grows")
state = None
outs = []
for t in range(n):
    out, state = causal_linear_attention_recurrent(q[:, :, t : t + 1], k[:, :, t : t + 1], v[:, :, t : t + 1], state)
    outs.append(out)
    if t in (0, 4, 9):
        print(f"  t={t}: S {tuple(state[0].shape)}, z {tuple(state[1].shape)}, |S| = {state[0].norm().item():.3f}")
rec = torch.cat(outs, dim=2)
print(f"max |recurrent - parallel| = {(rec - causal_linear_attention_parallel(q, k, v)).abs().max().item():.1e}")

print("\nHow far is elu+1 from softmax? It is a different kernel, not an approximation:")
qr, kr, vr = torch.randn(1, 1, 64, 16), torch.randn(1, 1, 64, 16), torch.randn(1, 1, 64, 16)
exact = torch.softmax(qr @ kr.transpose(-2, -1) / 4, -1) @ vr
elu = linear_attention(elu_feature_map(qr), elu_feature_map(kr), vr)
print(f"  mean |elu+1 - softmax| = {(elu - exact).abs().mean().item():.3f}, output std = {exact.std().item():.3f}")

print("\nPerformer random features do approximate softmax, and improve with m (advanced):")
print("     m   mean |approx - softmax|")
for m in [16, 64, 256, 1024, 4096]:
    errs = []
    for seed in range(8):
        w = torch.randn(m, 16, generator=torch.Generator().manual_seed(seed))
        approx = linear_attention(performer_features(qr * 0.5, w), performer_features(kr * 0.5, w), vr)
        target = torch.softmax((qr * 0.5) @ (kr * 0.5).transpose(-2, -1) / math.sqrt(16), -1) @ vr
        errs.append((approx - target).abs().mean().item())
    print(f"  {m:>5}   {sum(errs) / len(errs):.4f}")
