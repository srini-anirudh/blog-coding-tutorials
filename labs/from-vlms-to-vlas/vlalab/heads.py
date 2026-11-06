"""Three action heads on the same small MLP trunk: regression, per-dimension bins, and flow matching."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .tokenizers import UniformBinner


def mlp(inp: int, out: int, width: int = 128, depth: int = 3) -> nn.Sequential:
    layers, d = [], inp
    for _ in range(depth - 1):
        layers += [nn.Linear(d, width), nn.GELU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, out))


class RegressionHead(nn.Module):
    """Predict the action directly. Trained with MSE, so it predicts the mean of the demonstrated actions."""

    def __init__(self, obs_dim: int, act_dim: int, width: int = 128):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, width)

    def loss(self, obs, act):
        return F.mse_loss(self.net(obs), act)

    @torch.no_grad()
    def act(self, obs, generator=None):
        return self.net(obs)


class BinHead(nn.Module):
    """A 256-way softmax per action dimension, trained with cross-entropy, sampled at temperature 1.

    The dimensions are sampled independently, which is what "handles multimodal demos per dimension" means.
    """

    def __init__(self, obs_dim: int, act_dim: int, binner: UniformBinner, width: int = 128):
        super().__init__()
        self.binner, self.act_dim, self.n = binner, act_dim, binner.n_bins
        self.net = mlp(obs_dim, act_dim * self.n, width)

    def logits(self, obs):
        return self.net(obs).view(-1, self.act_dim, self.n)

    def loss(self, obs, act):
        target = torch.as_tensor(self.binner.encode(act.numpy()))
        return F.cross_entropy(self.logits(obs).reshape(-1, self.n), target.reshape(-1))

    @torch.no_grad()
    def act(self, obs, generator=None):
        probs = self.logits(obs).softmax(-1).reshape(-1, self.n)
        bins = torch.multinomial(probs, 1, generator=generator).view(-1, self.act_dim)
        return torch.as_tensor(self.binner.decode(bins.numpy()), dtype=torch.float32)


def sample_tau(n: int, s: float = 0.999, generator=None) -> torch.Tensor:
    """pi0's flow-matching time: u ~ Beta(1.5, 1), tau = s * (1 - u). Most mass sits near tau = 0 (heavy noise)."""
    u = torch.distributions.Beta(1.5, 1.0).sample((n,))
    return s * (1 - u)


class FlowHead(nn.Module):
    """Flow matching over an action vector (or a flattened chunk).

    Training: A_tau = tau * A + (1 - tau) * eps, and the net regresses the velocity A - eps.
    Inference: start from noise at tau = 0 and take `steps` Euler steps to tau = 1.
    """

    def __init__(self, obs_dim: int, act_dim: int, width: int = 256, depth: int = 4, n_freq: int = 4,
                 scale: torch.Tensor | None = None):
        super().__init__()
        self.act_dim, self.n_freq = act_dim, n_freq
        self.net = mlp(obs_dim + act_dim + 2 * n_freq, act_dim, width, depth)
        # Actions are learned in a normalised space; `scale` maps it back.
        self.register_buffer("scale", torch.ones(act_dim) if scale is None else scale)

    def time_features(self, tau):
        """sin and cos of tau at a few octaves. With raw tau as one input, a small MLP blurs the two modes."""
        e = tau[:, None] * math.pi * 2.0 ** torch.arange(self.n_freq, dtype=tau.dtype)[None]
        return torch.cat([torch.sin(e), torch.cos(e)], -1)

    def velocity(self, obs, a_tau, tau):
        return self.net(torch.cat([obs, a_tau, self.time_features(tau)], -1))

    def loss(self, obs, act):
        a = act / self.scale
        eps = torch.randn_like(a)
        tau = sample_tau(len(a))
        a_tau = tau[:, None] * a + (1 - tau[:, None]) * eps
        return F.mse_loss(self.velocity(obs, a_tau, tau), a - eps)

    @torch.no_grad()
    def act(self, obs, steps: int = 10, generator=None):
        a = torch.randn(len(obs), self.act_dim, generator=generator)
        for i in range(steps):
            tau = torch.full((len(obs),), i / steps)
            a = a + self.velocity(obs, a, tau) / steps
        return a * self.scale


def train(head: nn.Module, obs: np.ndarray, act: np.ndarray, steps: int = 3000, batch: int = 256,
          lr: float = 2e-3, seed: int = 0) -> list[float]:
    """AdamW with cosine decay on random minibatches. Returns the loss every 100 steps."""
    torch.manual_seed(seed)
    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(act)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    log = []
    for s in range(steps):
        idx = torch.randint(0, len(obs_t), (batch,))
        loss = head.loss(obs_t[idx], act_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 100 == 0 or s == steps - 1:
            log.append(float(loss.detach()))
    return log
