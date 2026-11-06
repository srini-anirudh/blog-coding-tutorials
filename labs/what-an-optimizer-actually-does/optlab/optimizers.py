"""Every optimizer in the post, written as the update rule it is.

All of them fit one template: theta <- theta - lr * P^-1 m.
  m  is the gradient smoothed in time (none, an accumulator or an EMA).
  P  is the metric: identity (SGD), a diagonal of gradient scales (Adagrad, RMSProp, Adam),
     a rank-1 diagonal (Adafactor), or "set every singular value to 1" (Muon).

Each class keeps its state in `self.state[p]` so `state_bytes()` can count what it costs.
C1 (c01-pytorch/scratchnet/optim.py) builds plain SGD and AdamW and checks them against torch.
This file adds the variants the post compares: Nesterov, the EMA convention, Adagrad, RMSProp,
coupled L2, the epsilon placement, Lion, Adafactor, 8-bit AdamW and Muon.
"""

from __future__ import annotations

import math

import torch

from .newton_schulz import newton_schulz
from .quant import BlockQuantized


class Optimizer:
    def __init__(self, params, lr: float) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.state: dict[torch.Tensor, dict] = {}

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def state_bytes(self) -> int:
        """Bytes held by optimizer state tensors (step counters are not counted)."""
        total = 0
        for s in self.state.values():
            for v in s.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
                elif isinstance(v, BlockQuantized):
                    total += v.nbytes()
        return total

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                self.update(p, p.grad, self.state.setdefault(p, {}))

    def update(self, p, g, s) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with optional momentum.

    ema=False is the accumulator form (PyTorch):  m = beta m + g
    ema=True  is the EMA form (Adam, Lion, Muon): m = beta m + (1 - beta) g
    For a constant gradient the first converges to g / (1 - beta), the second to g.
    Nesterov evaluates the step at the look-ahead point: d = g + beta m (PyTorch's form).
    """

    def __init__(self, params, lr, momentum=0.0, nesterov=False, ema=False, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum, self.nesterov, self.ema, self.wd = momentum, nesterov, ema, weight_decay

    def update(self, p, g, s):
        if self.wd:
            g = g + self.wd * p
        d = g
        if self.momentum:
            b = self.momentum
            if "m" not in s:
                s["m"] = (1 - b) * g if self.ema else g.clone()
            else:
                s["m"].mul_(b).add_(g, alpha=(1 - b) if self.ema else 1.0)
            d = g + b * s["m"] if self.nesterov else s["m"]
        p.add_(d, alpha=-self.lr)


class Adagrad(Optimizer):
    """v is a running sum of g^2, so the step size decays like 1 / sqrt(t)."""

    def __init__(self, params, lr, eps=1e-10):
        super().__init__(params, lr)
        self.eps = eps

    def update(self, p, g, s):
        s.setdefault("v", torch.zeros_like(p)).addcmul_(g, g)
        p.addcdiv_(g, s["v"].sqrt().add_(self.eps), value=-self.lr)


class RMSProp(Optimizer):
    """Adagrad with the sum replaced by an EMA: a window of 1 / (1 - beta2) steps."""

    def __init__(self, params, lr, beta2=0.99, eps=1e-8):
        super().__init__(params, lr)
        self.beta2, self.eps = beta2, eps

    def update(self, p, g, s):
        s.setdefault("v", torch.zeros_like(p)).mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
        p.addcdiv_(g, s["v"].sqrt().add_(self.eps), value=-self.lr)


class Adam(Optimizer):
    """Adam, with every switch the post discusses.

    decoupled=True  is AdamW: p <- p - lr wd p, outside the adaptive denominator.
    decoupled=False is Adam + L2: g <- g + wd p, so the decay is divided by sqrt(v) too.
    eps_inside=False is m / (sqrt(v) + eps); True is m / sqrt(v + eps).
    bias_correction=False drops the 1 / (1 - beta^t) factors.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 decoupled=True, bias_correction=True, eps_inside=False):
        super().__init__(params, lr)
        self.betas, self.eps, self.wd = betas, eps, weight_decay
        self.decoupled, self.bias_correction, self.eps_inside = decoupled, bias_correction, eps_inside

    def direction(self, p, g, s):
        """Update the moments and return m_hat / (sqrt(v_hat) + eps): the step before lr."""
        b1, b2 = self.betas
        if not s:
            s.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
        s["t"] += 1
        s["m"].lerp_(g, 1 - b1)
        s["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
        m, v = s["m"], s["v"]
        if self.bias_correction:
            m, v = m / (1 - b1 ** s["t"]), v / (1 - b2 ** s["t"])
        return m / (v + self.eps).sqrt() if self.eps_inside else m / (v.sqrt() + self.eps)

    def update(self, p, g, s):
        if self.wd and not self.decoupled:
            g = g + self.wd * p
        u = self.direction(p, g, s)
        if self.wd and self.decoupled:
            p.mul_(1 - self.lr * self.wd)
        p.add_(u, alpha=-self.lr)


class Lion(Optimizer):
    """u = sign(b1 m + (1 - b1) g); p <- p - lr (u + wd p); m <- b2 m + (1 - b2) g. One state tensor."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, lr)
        self.betas, self.wd = betas, weight_decay

    def update(self, p, g, s):
        b1, b2 = self.betas
        m = s.setdefault("m", torch.zeros_like(p))
        u = torch.sign(m * b1 + g * (1 - b1))
        p.mul_(1 - self.lr * self.wd).add_(u, alpha=-self.lr)
        m.lerp_(g, 1 - b2)


class Adafactor(Optimizer):
    """Factored second moment, no first moment (beta1 = 0), RMS update clipping.

    For a matrix, keep row sums R (m values) and column sums C (n values) of the EMA of g^2,
    and rebuild V_hat = R C^T / sum(R). Vectors keep a full v. A fixed lr and beta2 are used
    here instead of the paper's relative step size and beta2 schedule.
    """

    def __init__(self, params, lr=1e-2, beta2=0.999, eps=1e-30, clip=1.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta2, self.eps, self.clip, self.wd = beta2, eps, clip, weight_decay

    def update(self, p, g, s):
        b2 = self.beta2
        g2 = g * g + self.eps
        if p.ndim == 2:
            s.setdefault("r", torch.zeros(p.shape[0])).mul_(b2).add_(g2.mean(1), alpha=1 - b2)
            s.setdefault("c", torch.zeros(p.shape[1])).mul_(b2).add_(g2.mean(0), alpha=1 - b2)
            v = factored_v(s["r"], s["c"])
        else:
            v = s.setdefault("v", torch.zeros_like(p)).mul_(b2).add_(g2, alpha=1 - b2)
        s["t"] = s.get("t", 0) + 1
        u = g / (v / (1 - b2 ** s["t"])).sqrt()
        u = u / max(1.0, rms(u).item() / self.clip)
        if self.wd:
            p.mul_(1 - self.lr * self.wd)
        p.add_(u, alpha=-self.lr)


class AdamW8bit(Adam):
    """AdamW whose m and v live as 8-bit codes with one fp32 scale per block.

    Each step dequantizes, runs the ordinary Adam update in fp32, and quantizes back.
    `block` = None means one scale for the whole tensor (the naive version that outliers break).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 block: int | None = 2048, code: str = "log"):
        super().__init__(params, lr, betas, eps, weight_decay)
        self.block, self.code = block, code

    def update(self, p, g, s):
        inner = {}
        if s:
            inner = {"t": s["t"], "m": s["m"].dequantize(), "v": s["v"].dequantize()}
        u = self.direction(p, g, inner)
        s["t"] = inner["t"]
        s["m"] = BlockQuantized(inner["m"], self.block, self.code, signed=True)
        s["v"] = BlockQuantized(inner["v"], self.block, self.code, signed=False)
        if self.wd:
            p.mul_(1 - self.lr * self.wd)
        p.add_(u, alpha=-self.lr)


class Muon(Optimizer):
    """Momentum, then replace the momentum matrix with (nearly) the closest semi-orthogonal matrix.

    d = NewtonSchulz(nesterov momentum), rescaled by rms_scale * sqrt(max(m, n)) so that
    RMS(d) = rms_scale (0.2, the Moonshot recipe). Only for 2-D parameters. ns_dtype is bf16 as in
    Muon; CPU runs here pass float32, because bf16 matmuls are slow on CPU.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 rms_scale=0.2, weight_decay=0.0, ns_dtype=torch.bfloat16):
        super().__init__(params, lr)
        self.momentum, self.nesterov, self.ns_steps, self.ns_dtype = momentum, nesterov, ns_steps, ns_dtype
        self.rms_scale, self.wd = rms_scale, weight_decay

    def update(self, p, g, s):
        b = self.momentum
        m = s.setdefault("m", torch.zeros_like(p)).lerp_(g, 1 - b)
        d = g.lerp(m, b) if self.nesterov else m
        mat = d.reshape(d.shape[0], -1) if d.ndim > 1 else d.reshape(1, -1)
        o = newton_schulz(mat, self.ns_steps, dtype=self.ns_dtype).to(p.dtype).reshape(p.shape)
        if self.wd:
            p.mul_(1 - self.lr * self.wd)
        p.add_(o, alpha=-self.lr * self.rms_scale * math.sqrt(max(mat.shape)))


class Hybrid:
    """Several optimizers behind one step(): Muon for hidden matrices, AdamW for the rest."""

    def __init__(self, *opts: Optimizer) -> None:
        self.opts = opts
        self.params = [p for o in opts for p in o.params]

    def step(self) -> None:
        for o in self.opts:
            o.step()

    def zero_grad(self) -> None:
        for o in self.opts:
            o.zero_grad()

    def state_bytes(self) -> int:
        return sum(o.state_bytes() for o in self.opts)


def factored_v(r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Adafactor's rank-1 rebuild: the outer product of row and column statistics over their total."""
    return torch.outer(r, c) / r.mean()


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt()
