"""Hand-written VJPs for every layer in a transformer, checked against torch.autograd.

Then the structure the post points out: LayerNorm's gradient is orthogonal to 1 and to x_hat, RMSNorm's only to
x_hat, softmax's sums to zero against p, GELU's derivative dips below zero, and the dot-product test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from autogradlab import vjp  # noqa: E402
from autogradlab.report import Report  # noqa: E402

rep = Report("04_layer_vjps")
torch.manual_seed(0)
dt = torch.float64
B, n, m = 8, 16, 12


def leaf(*shape):
    return torch.randn(*shape, dtype=dt, requires_grad=True)


def autograd_grads(out, inputs, g):
    return torch.autograd.grad(out, inputs, g)


rows = []


def check(name, ours, theirs):
    err = max((a - b).abs().max().item() for a, b in zip(ours, theirs))
    rows.append([name, len(ours), f"{err:.1e}"])


X, W, b = leaf(B, n), leaf(n, m), leaf(m)
U, s = vjp.linear_forward(X, W, b)
gU = torch.randn_like(U)
check("Linear (dX, dW, db)", vjp.linear_backward(s, gU), autograd_grads(X @ W + b, (X, W, b), gU))

x = leaf(B, n)
gy = torch.randn(B, n, dtype=dt)
y, s = vjp.relu_forward(x)
check("ReLU", [vjp.relu_backward(s, gy)], autograd_grads(F.relu(x), (x,), gy))
y, s = vjp.gelu_forward(x)
check("GELU (exact)", [vjp.gelu_backward(s, gy)], autograd_grads(F.gelu(x), (x,), gy))

gamma, beta = leaf(n), leaf(n)
y, s_ln = vjp.layernorm_forward(x, gamma, beta)
ref = F.layer_norm(x, (n,), gamma, beta, eps=1e-5)
check("LayerNorm (dx, dgamma, dbeta)", vjp.layernorm_backward(s_ln, gy), autograd_grads(ref, (x, gamma, beta), gy))

y, s_rms = vjp.rmsnorm_forward(x, gamma)
ref = F.rms_norm(x, (n,), gamma, eps=1e-6)
check("RMSNorm (dx, dgamma)", vjp.rmsnorm_backward(s_rms, gy), autograd_grads(ref, (x, gamma), gy))

z = leaf(B, n)
p, s_sm = vjp.softmax_forward(z)
check("Softmax", [vjp.softmax_backward(s_sm, gy)], autograd_grads(torch.softmax(z, -1), (z,), gy))

target = torch.randint(0, n, (B,))
L, s_ce = vjp.cross_entropy_forward(z, target)
check("Cross-entropy on logits", [vjp.cross_entropy_backward(s_ce)], autograd_grads(F.cross_entropy(z, target), (z,),
                                                                                    torch.tensor(1.0, dtype=dt)))

# Residual around a tanh layer: dx = dy + J_F^T dy.
Wr = torch.randn(n, n, dtype=dt)
gF = (gy * (1 - torch.tanh(x @ Wr) ** 2)) @ Wr.T
check("Residual x + tanh(x W)", [vjp.residual_backward(gy, gF)], autograd_grads(x + torch.tanh(x @ Wr), (x,), gy))

Q, K, V = leaf(2, 10, 8), leaf(2, 10, 8), leaf(2, 10, 8)
O, s_at = vjp.attention_forward(Q, K, V)
gO = torch.randn_like(O)
ref = autograd_grads(F.scaled_dot_product_attention(Q, K, V), (Q, K, V), gO)
check("Attention (dQ, dK, dV)", vjp.attention_backward(s_at, gO), ref)
check("Attention, P recomputed", autograd_grads(vjp.AttentionRecompute.apply(Q, K, V), (Q, K, V), gO), ref)

rep("1. Hand-written VJP against torch.autograd (float64, random upstream gradients):")
rep.table(["layer", "grads", "max abs error"], rows)

rep("\n2. Invariance in the forward pass shows up as a projection in the backward pass:")
def dots(g, xhat):
    return f"{g.sum(-1).abs().max().item():.1e}", f"{(g * xhat).sum(-1).abs().max().item():.1e}"


table = []
for eps in [1e-5, 0.0]:
    _, s1 = vjp.layernorm_forward(x, gamma, beta, eps=eps)
    _, s2 = vjp.rmsnorm_forward(x, gamma, eps=eps)
    table.append([f"LayerNorm dx, eps={eps:g}", *dots(vjp.layernorm_backward(s1, gy)[0], s1[0])])
    table.append([f"RMSNorm dx, eps={eps:g}", *dots(vjp.rmsnorm_backward(s2, gy)[0], s2[0])])
table.append(["Softmax dz", f"{vjp.softmax_backward(s_sm, gy).sum(-1).abs().max().item():.1e}", "-"])
rep.table(["gradient", "largest |dot with 1| per row", "largest |dot with x_hat| per row"], table)
rep("LayerNorm ignores shifts and scales of x, so dx has no component along 1 or x_hat.")
rep("RMSNorm ignores only scale, so dx is orthogonal to x_hat but not to 1.")
rep("Softmax ignores a constant added to every logit, so every row of dz sums to zero.")
rep("The x_hat projection is exact only at eps = 0. With eps > 0 the layer is not quite scale-invariant, and the")
rep("dot product is small but not zero. The shift invariance of LayerNorm does not depend on eps.")

xs = torch.linspace(-3, 1, 40001, dtype=dt)
d = vjp.gelu_grad(xs)
zero = xs[(d[1:].sign() != d[:-1].sign()).nonzero()[0, 0]].item()
rep(f"\n3. GELU derivative: zero at x = {zero:.3f} (where GELU itself is lowest), negative below that, and lowest")
rep(f"   ({d.min().item():.4f}) at x = {xs[d.argmin()].item():.3f}. Units that were nearly off still pass a little gradient.")

rep("\n4. Dot-product test <u, J v> = <J^T u, v> with our VJP and torch.func.jvp for J v:")
rows = []
for name, f, bwd, shape in [
    ("LayerNorm", lambda t: vjp.layernorm_forward(t, gamma.detach(), beta.detach())[0],
     lambda t, u: vjp.layernorm_backward(vjp.layernorm_forward(t, gamma.detach(), beta.detach())[1], u)[0], (B, n)),
    ("Softmax", lambda t: torch.softmax(t, -1), lambda t, u: vjp.softmax_backward((torch.softmax(t, -1),), u), (B, n)),
    ("GELU", lambda t: vjp.gelu_forward(t)[0], lambda t, u: vjp.gelu_backward((t,), u), (B, n)),
]:
    t = torch.randn(shape, dtype=dt)
    v, u = torch.randn(shape, dtype=dt), torch.randn(shape, dtype=dt)
    _, Jv = torch.func.jvp(f, (t,), (v,))
    rows.append([name, f"{(u * Jv).sum().item():.12f}", f"{(bwd(t, u) * v).sum().item():.12f}"])
rep.table(["layer", "<u, J v>", "<J^T u, v>"], rows)
rep.save()
