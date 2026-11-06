"""Custom backward rules: a correct linear layer, a buggy one, a straight-through estimator, and a dot-product test."""

from __future__ import annotations

import torch


class MyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)  # not ctx.x = x: save_for_backward is what the version counter checks
        return x @ w

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        return gy @ w.T, x.T @ gy  # one gradient per forward input


class BuggyLinear(MyLinear):
    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        return gy @ w.T, (x.T @ gy).T.reshape(w.shape)  # transposed dW, reshaped to fit: shapes alone do not catch it


def dot_test(fn, x, w, seed=0):
    """<u, J v> against <J^T u, v>, with J v from a central difference in float64."""
    gen = torch.Generator().manual_seed(seed)
    vx, vw = torch.randn(x.shape, generator=gen, dtype=x.dtype), torch.randn(w.shape, generator=gen, dtype=x.dtype)
    eps = 1e-6
    Jv = (fn(x + eps * vx, w + eps * vw) - fn(x - eps * vx, w - eps * vw)) / (2 * eps)
    u = torch.randn(Jv.shape, generator=gen, dtype=x.dtype)
    xx, ww = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    gx, gw = torch.autograd.grad(fn(xx, ww), (xx, ww), u)
    return (u * Jv).sum().item(), ((gx * vx).sum() + (gw * vw).sum()).item()


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, gy):
        return gy  # pretend round is the identity
