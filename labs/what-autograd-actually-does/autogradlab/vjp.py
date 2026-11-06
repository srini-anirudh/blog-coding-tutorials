"""The backward pass of every layer in a transformer, written by hand as a VJP.

Each `*_forward` returns (output, saved), where `saved` is what the backward rule needs. Each `*_backward`
takes (saved, upstream gradient) and returns the downstream gradients. No Jacobian is ever built.
Rows are the last dimension; everything before it is batch.
"""

from __future__ import annotations

import math

import torch

# Linear: U = X W + b


def linear_forward(X, W, b):
    return X @ W + b, (X, W)


def linear_backward(saved, gU):
    X, W = saved
    return gU @ W.T, X.T @ gU, gU.sum(0)  # dX, dW, db


# ReLU: J is diagonal, so the VJP is a mask.


def relu_forward(x):
    y = x.clamp_min(0)
    return y, (y > 0,)  # only the sign pattern is needed


def relu_backward(saved, gy):
    (mask,) = saved
    return gy * mask


# GELU (exact, erf form): dy/dx = Phi(x) + x phi(x)


def gelu_grad(x):
    Phi = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    phi = torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return Phi + x * phi


def gelu_forward(x):
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2))), (x,)


def gelu_backward(saved, gy):
    (x,) = saved
    return gy * gelu_grad(x)


# LayerNorm: the backward pass removes the mean of g and the component of g along x_hat.


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdim=True)
    sigma = ((x - mu).pow(2).mean(-1, keepdim=True) + eps).sqrt()
    xhat = (x - mu) / sigma
    return gamma * xhat + beta, (xhat, sigma, gamma)


def layernorm_backward(saved, gy):
    xhat, sigma, gamma = saved
    g = gy * gamma
    gx = (g - g.mean(-1, keepdim=True) - xhat * (g * xhat).mean(-1, keepdim=True)) / sigma
    rows = tuple(range(gy.dim() - 1))
    return gx, (gy * xhat).sum(rows), gy.sum(rows)  # dx, dgamma, dbeta


# RMSNorm: one projection instead of two, because there is no mean to remove.


def rmsnorm_forward(x, gamma, eps=1e-6):
    rms = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    xhat = x / rms
    return gamma * xhat, (xhat, rms, gamma)


def rmsnorm_backward(saved, gy):
    xhat, rms, gamma = saved
    g = gy * gamma
    gx = (g - xhat * (g * xhat).mean(-1, keepdim=True)) / rms
    return gx, (gy * xhat).sum(tuple(range(gy.dim() - 1)))


# Softmax: J = diag(p) - p p^T, applied in O(n).


def softmax_forward(z):
    p = torch.softmax(z, -1)
    return p, (p,)


def softmax_backward(saved, gp):
    (p,) = saved
    return p * (gp - (p * gp).sum(-1, keepdim=True))


# Cross-entropy on logits, fused: dz = (p - y) / batch.


def cross_entropy_forward(z, target):
    logp = torch.log_softmax(z, -1)
    loss = -logp.gather(-1, target[:, None]).mean()
    return loss, (logp.exp(), target)


def cross_entropy_backward(saved, gL=1.0):
    p, target = saved
    y = torch.nn.functional.one_hot(target, p.shape[-1]).to(p.dtype)
    return gL * (p - y) / p.shape[0]


# Residual: y = x + F(x), so dx = dy + J_F^T dy. The skip contributes the upstream gradient unchanged.


def residual_backward(gy, gx_through_F):
    return gy + gx_through_F


# Attention: S = Q K^T / sqrt(d), P = softmax(S), O = P V.


def attention_forward(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.transpose(-1, -2) / math.sqrt(d)
    P = torch.softmax(S, -1)
    return P @ V, (Q, K, V, P)


def attention_backward(saved, gO):
    Q, K, V, P = saved
    d = Q.shape[-1]
    gV = P.transpose(-1, -2) @ gO
    gP = gO @ V.transpose(-1, -2)
    gS = P * (gP - (P * gP).sum(-1, keepdim=True))
    gQ = gS @ K / math.sqrt(d)
    gK = gS.transpose(-1, -2) @ Q / math.sqrt(d)
    return gQ, gK, gV


class AttentionRecompute(torch.autograd.Function):
    """Attention that saves only Q, K, V and rebuilds P in the backward pass, as FlashAttention does.

    The saved bill drops from O(N^2) to O(N d). The backward pass now has five matmuls instead of four.
    """

    @staticmethod
    def forward(ctx, Q, K, V):
        O, _ = attention_forward(Q, K, V)
        ctx.save_for_backward(Q, K, V)
        return O

    @staticmethod
    def backward(ctx, gO):
        Q, K, V = ctx.saved_tensors
        P = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1]), -1)  # the recomputed matmul
        return attention_backward((Q, K, V, P), gO)


class MatmulCounter(torch.utils._python_dispatch.TorchDispatchMode):
    """Count matrix multiplications that reach the dispatcher, split by forward and backward phase."""

    MATMULS = {"mm", "bmm", "addmm", "matmul", "baddbmm"}

    def __init__(self):
        super().__init__()
        self.counts: dict[str, int] = {}
        self.phase = "forward"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket.__name__ in self.MATMULS:
            self.counts[self.phase] = self.counts.get(self.phase, 0) + 1
        return func(*args, **(kwargs or {}))
