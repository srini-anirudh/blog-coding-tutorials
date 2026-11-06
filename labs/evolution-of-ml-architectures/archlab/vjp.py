"""The post's "backward pass, derived", one function per formula. Each takes the forward inputs and the
output adjoint (Y_bar = dL/dY) and returns input adjoints. No autograd inside: the tests and
scripts/07_backward_derived.py compare every one with torch.autograd.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def linear(X, W, Yb):
    """Y = X W + b  ->  X_bar = Y_bar W^T,  W_bar = X^T Y_bar,  b_bar = 1^T Y_bar."""
    return Yb @ W.T, X.T @ Yb, Yb.sum(0)


ACT_DERIVATIVES = {
    "relu": lambda z: (z > 0).to(z.dtype),
    "tanh": lambda z: 1 - torch.tanh(z) ** 2,
    "sigmoid": lambda z: torch.sigmoid(z) * (1 - torch.sigmoid(z)),
    "gelu": lambda z: 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi),
    "silu": lambda z: torch.sigmoid(z) * (1 + z * (1 - torch.sigmoid(z))),
}


def softmax(p, pb):
    """z_bar = p * (p_bar - p^T p_bar), row-wise."""
    return p * (pb - (p * pb).sum(-1, keepdim=True))


def softmax_cross_entropy(z, y):
    """Mean cross-entropy over the batch. z_bar = (p - onehot(y)) / B."""
    return (torch.softmax(z, -1) - F.one_hot(y, z.shape[-1]).to(z.dtype)) / z.shape[0]


def layernorm(x, gamma, yb, eps=1e-5):
    """x_bar = (1/sigma) [g - mean(g) - x_hat mean(g x_hat)], g = y_bar * gamma. Also gamma_bar, beta_bar."""
    mu = x.mean(-1, keepdim=True)
    sigma = torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
    xh = (x - mu) / sigma
    g = yb * gamma
    xb = (g - g.mean(-1, keepdim=True) - xh * (g * xh).mean(-1, keepdim=True)) / sigma
    return xb, (yb * xh).sum(0), yb.sum(0)


def rmsnorm(x, gamma, yb, eps=1e-6):
    """x_bar = (1/rms) [g - x_hat mean(g x_hat)]: one projection instead of two."""
    rms = torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps)
    xh = x / rms
    g = yb * gamma
    return (g - xh * (g * xh).mean(-1, keepdim=True)) / rms


def conv2d(X, K, Yb):
    """Y = correlate(X, K), stride 1, no padding.
    K_bar = correlate(X, Y_bar) summed over the batch; X_bar = full convolution of Y_bar with the flipped kernel."""
    Kb = F.conv2d(X.transpose(0, 1), Yb.transpose(0, 1)).transpose(0, 1)  # (C_out, C_in, k, k)
    k = K.shape[-1]
    Xb = F.conv2d(F.pad(Yb, (k - 1,) * 4), K.flip(-1, -2).transpose(0, 1))
    return Xb, Kb


def rnn_bptt(xs, h0, Wh, Wx, b, hbs):
    """Elman RNN h_t = tanh(W_h h_{t-1} + W_x x_t + b), loss adjoints hbs[t] = dL_t/dh_t.
    delta_t = hbs[t] + W_h^T (delta_{t+1} * (1 - h_{t+1}^2)). Returns W_h_bar, W_x_bar, h0_bar."""
    hs, h = [], h0
    for x in xs:
        h = torch.tanh(Wh @ h + Wx @ x + b)
        hs.append(h)
    Whb, Wxb = torch.zeros_like(Wh), torch.zeros_like(Wx)
    carry = torch.zeros_like(h0)  # W_h^T (delta_{t+1} * (1 - h_{t+1}^2))
    for t in reversed(range(len(xs))):
        delta = hbs[t] + carry
        da = delta * (1 - hs[t] ** 2)
        h_prev = hs[t - 1] if t > 0 else h0
        Whb += torch.outer(da, h_prev)
        Wxb += torch.outer(da, xs[t])
        carry = Wh.T @ da
    return Whb, Wxb, carry


def lstm_cell(f, i, g, o, c_prev, c, hb, cb_next, f_next):
    """One step of the LSTM adjoints from the post. Gates are post-activation values.
    dc_t = dh_t o_t (1 - tanh^2 c_t) + dc_{t+1} f_{t+1}; then the four gate pre-activation adjoints."""
    dc = hb * o * (1 - torch.tanh(c) ** 2) + cb_next * f_next
    do = hb * torch.tanh(c) * o * (1 - o)
    df = dc * c_prev * f * (1 - f)
    di = dc * g * i * (1 - i)
    dg = dc * i * (1 - g ** 2)
    return dc, df, di, dg, do


def attention(Q, K, V, Ob):
    """S = Q K^T / sqrt(d), P = softmax(S), O = P V. Returns Q_bar, K_bar, V_bar and the row term D two ways."""
    scale = 1 / math.sqrt(Q.shape[-1])
    P = torch.softmax(Q @ K.T * scale, -1)
    O = P @ V
    Vb = P.T @ Ob
    Pb = Ob @ V.T
    D_from_P = (Pb * P).sum(-1, keepdim=True)  # needs the N x N matrix
    D_from_O = (Ob * O).sum(-1, keepdim=True)  # needs only N x d tensors: the FlashAttention identity
    Sb = P * (Pb - D_from_O)
    return scale * Sb @ K, scale * Sb.T @ Q, Vb, D_from_P, D_from_O


def ssm_scan_backward(a, b, c, yb):
    """h_t = a_t h_{t-1} + b_t, y_t = c_t h_t (all elementwise, time on dim 0).
    lambda_t = c_t y_bar_t + a_{t+1} lambda_{t+1}: a linear recurrence run backward.
    Returns a_bar = lambda_t h_{t-1}, b_bar = lambda_t, c_bar = y_bar_t h_t."""
    T = a.shape[0]
    hs, h = [], torch.zeros_like(b[0])
    for t in range(T):
        h = a[t] * h + b[t]
        hs.append(h)
    lam = torch.zeros_like(b)
    nxt = torch.zeros_like(b[0])
    for t in reversed(range(T)):
        nxt = c[t] * yb[t] + (a[t + 1] * nxt if t + 1 < T else 0)
        lam[t] = nxt
    h_prev = torch.stack([torch.zeros_like(b[0])] + hs[:-1])
    return lam * h_prev, lam, yb * torch.stack(hs)


def moe_router(x, Wr, experts, top_k, yb):
    """y = sum_{k in top-K} g_k E_k(x), g = softmax over the chosen logits. g_bar_k = <y_bar, E_k(x)>,
    then through the softmax. Unchosen experts' logits get exactly zero gradient."""
    logits = x @ Wr
    idx = logits.topk(top_k).indices
    g = torch.softmax(logits[idx], -1)
    gb = torch.stack([(yb * experts[k](x)).sum() for k in idx.tolist()])
    lb = torch.zeros_like(logits)
    lb[idx] = softmax(g, gb)
    return lb
