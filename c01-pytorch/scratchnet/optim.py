"""Optimizers written as the update rules they are.

An optimizer holds references to parameters and some per-parameter state.
`step()` reads each `p.grad` and changes `p` in place. That is all.
These match torch.optim.SGD and torch.optim.AdamW to float precision.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


class ManualOptimizer:
    """The shared part: a parameter list, per-parameter state and zero_grad."""

    def __init__(self, params: Iterable[torch.nn.Parameter]) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.state: dict[torch.Tensor, dict[str, torch.Tensor | int]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if set_to_none:
                # Dropping the tensor frees memory, and the next backward
                # writes a fresh gradient instead of adding to zeros.
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_dict(self) -> dict:
        """State in parameter order, so it can be saved and loaded across processes."""
        return {"state": [self.state.get(p, {}) for p in self.params]}

    def load_state_dict(self, sd: dict) -> None:
        for p, s in zip(self.params, sd["state"]):
            if s:
                self.state[p] = {k: v.to(p.device) if torch.is_tensor(v) else v for k, v in s.items()}


class SGD(ManualOptimizer):
    """p <- p - lr * g, with optional momentum and (coupled) weight decay."""

    def __init__(self, params, lr: float, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        super().__init__(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay

    @torch.no_grad()  # the update itself must not be recorded in the autograd graph
    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g.add(p, alpha=self.weight_decay)  # L2 penalty folded into the gradient
            if self.momentum:
                state = self.state.setdefault(p, {})
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                else:
                    state["momentum_buffer"].mul_(self.momentum).add_(g)
                g = state["momentum_buffer"]
            p.add_(g, alpha=-self.lr)


class AdamW(ManualOptimizer):
    """Adam with decoupled weight decay (Loshchilov and Hutter, 2019).

    m <- b1 m + (1 - b1) g          first moment (a running mean of g)
    v <- b2 v + (1 - b2) g^2        second moment (a running mean of g^2)
    p <- p (1 - lr wd)              decay applied to p, not mixed into g
    p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params)
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay

    @torch.no_grad()
    def step(self) -> None:
        beta1, beta2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state.setdefault(p, {})
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)  # m: same shape as p
                state["exp_avg_sq"] = torch.zeros_like(p)  # v: same shape as p
            state["step"] += 1
            t = state["step"]
            m, v = state["exp_avg"], state["exp_avg_sq"]

            p.mul_(1 - self.lr * self.weight_decay)
            m.lerp_(g, 1 - beta1)  # m + (1 - b1)(g - m) == b1 m + (1 - b1) g
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            # Both moments start at zero, so early averages are biased toward
            # zero. Dividing by (1 - beta^t) removes that bias.
            bias1 = 1 - beta1**t
            bias2 = 1 - beta2**t
            denom = (v.sqrt() / math.sqrt(bias2)).add_(self.eps)
            p.addcdiv_(m, denom, value=-self.lr / bias1)


def grad_norm(params: Iterable[torch.Tensor]) -> torch.Tensor:
    """Global L2 norm of all gradients: sqrt(sum over params of ||g||^2)."""
    norms = [p.grad.detach().norm(2) for p in params if p.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms), 2)


@torch.no_grad()
def clip_grad_norm(params: Iterable[torch.Tensor], max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """Scale every gradient by min(1, max_norm / total_norm). Returns the norm before clipping."""
    params = [p for p in params if p.grad is not None]
    total = grad_norm(params)
    scale = (max_norm / (total + eps)).clamp(max=1.0)
    for p in params:
        p.grad.mul_(scale)
    return total
