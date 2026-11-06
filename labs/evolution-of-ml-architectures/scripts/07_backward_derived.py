"""The backward pass, derived: every hand-written formula from the post against torch.autograd (float64).

Also two forward-side claims that are really backward-pass claims: the 1/sqrt(d_k) keeps the softmax
Jacobian alive, and LayerNorm's gradient is orthogonal to 1 and to x_hat. Then mixed-precision loss
scaling and global-norm clipping.
"""

import _setup  # noqa: F401

import math

import torch
import torch.nn.functional as F
from torch import nn

from archlab import vjp
from archlab.env import start

start("07_backward_derived")
torch.set_default_dtype(torch.float64)


def grads(fn, *xs):
    xs = [x.detach().clone().requires_grad_(True) for x in xs]
    fn(*xs).backward()
    return [x.grad for x in xs]


def err(a, b):
    return (a - b).abs().max().item()


rows = []
g = torch.Generator().manual_seed(0)
r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731

X, W, b = r(8, 5), r(5, 3), r(3)
Yb = r(8, 3)
Xb, Wb, bb = vjp.linear(X, W, Yb)
aX, aW, ab = grads(lambda X, W, b: ((X @ W + b) * Yb).sum(), X, W, b)
rows.append(("linear: X_bar = Y_bar W^T, W_bar = X^T Y_bar, b_bar = 1^T Y_bar", max(err(Xb, aX), err(Wb, aW), err(bb, ab))))

z = r(50)
for name, dfn in vjp.ACT_DERIVATIVES.items():
    f = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "gelu": F.gelu, "silu": F.silu}[name]
    (az,) = grads(lambda z: f(z).sum(), z)
    rows.append((f"{name}' formula", err(dfn(z), az)))

zl, pb = r(4, 6), r(4, 6)
(az,) = grads(lambda z: (torch.softmax(z, -1) * pb).sum(), zl)
rows.append(("softmax: z_bar = p * (p_bar - p.p_bar)", err(vjp.softmax(torch.softmax(zl, -1), pb), az)))
y = torch.randint(0, 6, (4,), generator=g)
(az,) = grads(lambda z: F.cross_entropy(z, y), zl)
rows.append(("softmax + cross-entropy: z_bar = p - y", err(vjp.softmax_cross_entropy(zl, y), az)))

x, gamma, beta, yb = r(4, 16), r(16), r(16), r(4, 16)
xb, gb, bb2 = vjp.layernorm(x, gamma, yb)
ax, ag, abt = grads(lambda x, gm, bt: (F.layer_norm(x, (16,), gm, bt) * yb).sum(), x, gamma, beta)
rows.append(("LayerNorm three-term x_bar, gamma_bar, beta_bar", max(err(xb, ax), err(gb, ag), err(bb2, abt))))
xh = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
ortho = max((xb.sum(-1)).abs().max().item(), (xb * xh).sum(-1).abs().max().item())
rmsn = lambda x: x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * gamma  # noqa: E731
(ax,) = grads(lambda x: (rmsn(x) * yb).sum(), x)
rows.append(("RMSNorm two-term x_bar", err(vjp.rmsnorm(x, gamma, yb), ax)))

Xc, Kc = r(2, 3, 9, 9), r(4, 3, 3, 3)
Ycb = r(2, 4, 7, 7)
Xcb, Kcb = vjp.conv2d(Xc, Kc, Ycb)
aX, aK = grads(lambda X, K: (F.conv2d(X, K) * Ycb).sum(), Xc, Kc)
rows.append(("conv: K_bar = correlate(X, Y_bar); X_bar = full conv with flipped K", max(err(Xcb, aX), err(Kcb, aK))))
tconv = F.conv_transpose2d(Ycb, Kc)
rows.append(("conv: X_bar equals conv_transpose2d(Y_bar, K)", err(Xcb, tconv)))

T, d = 12, 6
xs, h0, Wh, Wx, bh = r(T, d), r(d), 0.5 * r(d, d), r(d, d), r(d)
hbs = r(T, d)


def rnn_loss(Wh, Wx, h0):
    h, loss = h0, 0
    for t in range(T):
        h = torch.tanh(Wh @ h + Wx @ xs[t] + bh)
        loss = loss + (h * hbs[t]).sum()
    return loss


mWh, mWx, mh0 = vjp.rnn_bptt(xs, h0, Wh, Wx, bh, hbs)
aWh, aWx, ah0 = grads(rnn_loss, Wh, Wx, h0)
rows.append(("BPTT: delta_t = local + W_h^T (delta_{t+1} (1 - h^2))", max(err(mWh, aWh), err(mWx, aWx), err(mh0, ah0))))

# One LSTM step, gates as leaves: check dc_t and the four gate adjoints.
fp, ip, gp, op, cp = r(d), r(d), r(d), r(d), r(d)
hb, cbn, fn = r(d), r(d), torch.sigmoid(r(d))


def lstm_loss(fp, ip, gp, op, cp):
    f, i, gg, o = torch.sigmoid(fp), torch.sigmoid(ip), torch.tanh(gp), torch.sigmoid(op)
    c = f * cp + i * gg
    h = o * torch.tanh(c)
    return (h * hb).sum() + (c * fn * cbn).sum()  # the next step sees c through f_{t+1}


f, i, gg, o = torch.sigmoid(fp), torch.sigmoid(ip), torch.tanh(gp), torch.sigmoid(op)
dc, df, di, dg, do = vjp.lstm_cell(f, i, gg, o, cp, f * cp + i * gg, hb, cbn, fn)
af, ai, ag2, ao, acp = grads(lstm_loss, fp, ip, gp, op, cp)
rows.append(("LSTM: dc_t, df, di, dg, do and dc_{t-1} = dc_t f_t",
             max(err(df, af), err(di, ai), err(dg, ag2), err(do, ao), err(dc * f, acp))))

Q, K, V, Ob = r(10, 8), r(10, 8), r(10, 8), r(10, 8)
mQ, mK, mV, D1, D2 = vjp.attention(Q, K, V, Ob)
aQ, aK, aV = grads(lambda Q, K, V: (torch.softmax(Q @ K.T / math.sqrt(8), -1) @ V * Ob).sum(), Q, K, V)
rows.append(("attention: Q_bar, K_bar, V_bar", max(err(mQ, aQ), err(mK, aK), err(mV, aV))))
rows.append(("attention: rowsum(P_bar * P) = rowsum(O_bar * O)", err(D1, D2)))

a_s, b_s, c_s, yb_s = torch.sigmoid(r(20, 4)), r(20, 4), r(20, 4), r(20, 4)


def scan_loss(a, b, c):
    h, loss = torch.zeros(4), 0
    for t in range(20):
        h = a[t] * h + b[t]
        loss = loss + (c[t] * h * yb_s[t]).sum()
    return loss


ma, mb, mc = vjp.ssm_scan_backward(a_s, b_s, c_s, yb_s)
aa, ab3, ac = grads(scan_loss, a_s, b_s, c_s)
rows.append(("SSM: adjoint is a backward linear scan", max(err(ma, aa), err(mb, ab3), err(mc, ac))))
A, dt = -torch.rand(4) - 0.5, r(4)
(adt,) = grads(lambda dt: torch.exp(F.softplus(dt) * A).sum(), dt)
manual = A * torch.exp(F.softplus(dt) * A) * torch.sigmoid(dt)
rows.append(("SSM: d exp(Delta A)/d dt = A exp(Delta A) sigmoid(dt)", err(manual, adt)))

xm, Wr = r(6), r(6, 5)
experts = [nn.Linear(6, 6) for _ in range(5)]
ym = r(6)


def _moe(x, Wr, experts, yb):
    logits = x @ Wr
    idx = logits.topk(2).indices
    gsel = torch.softmax(logits[idx], -1)
    return (sum(gk * experts[k](x) for gk, k in zip(gsel, idx.tolist())) * yb).sum()


(aW,) = grads(lambda Wr: _moe(xm, Wr, experts, ym), Wr)
lb = vjp.moe_router(xm, Wr, experts, 2, ym)
rows.append(("MoE: router logit adjoint, top-2 of 5", err(torch.outer(xm, lb), aW)))
unchosen = (aW.abs().sum(0) == 0).sum().item()

print(f"{'formula':>72}   max |manual - autograd|")
for name, e in rows:
    print(f"{name:>72}   {e:.1e}")
print()
print(f"LayerNorm x_bar . 1 and x_bar . x_hat per row: max |.| = {ortho:.1e} (eps = 1e-5 keeps it from 0)")
print(f"MoE: router columns with exactly zero gradient: {unchosen} of 5 (the 3 experts not chosen)")
print()

torch.set_default_dtype(torch.float32)
print("Why 1/sqrt(d_k): q, k with unit-variance entries, d_k = 128, 64 keys")
gen = torch.Generator().manual_seed(1)
q, k = torch.randn(2000, 128, generator=gen), torch.randn(64, 128, generator=gen)
for label, scale in [("unscaled", 1.0), ("1/sqrt(d_k)", 1 / math.sqrt(128))]:
    s = q @ k.T * scale
    p = torch.softmax(s, -1)
    jac = (p * (1 - p)).sum(-1).mean().item()  # trace of diag(p) - p p^T
    print(f"   {label:>12}: logit std {s.std().item():5.2f}   max p {p.max(-1).values.mean().item():.3f}"
          f"   trace of softmax Jacobian {jac:.4f}")
print(f"   sqrt(128) = {math.sqrt(128):.2f}")
print()

print("Sigmoid derivative bound: max sigma' = 1/4, six layers: (1/4)^6 =", f"{0.25 ** 6:.2e}")
print()

print("Loss scaling in fp16: a gradient of 1e-8")
tiny = torch.tensor(1e-8)
print(f"   cast to fp16: {tiny.half().item():.1e}    scaled by 2^16, cast, unscaled: "
      f"{(tiny * 2 ** 16).half().float().item() / 2 ** 16:.2e}")
print("Clipping a gradient (3, 0.1) to norm 1")
gv = torch.tensor([3.0, 0.1])
glob = gv * min(1, 1 / gv.norm().item())
elem = gv.clamp(-1 / math.sqrt(2), 1 / math.sqrt(2))
cos = lambda u: (u @ gv / (u.norm() * gv.norm())).item()  # noqa: E731
print(f"   global norm: {[round(v, 4) for v in glob.tolist()]}  cosine to original {cos(glob):.4f}")
print(f"   elementwise to +-0.707: {[round(v, 4) for v in elem.tolist()]}  cosine to original {cos(elem):.4f}")
