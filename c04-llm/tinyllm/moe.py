"""A minimal mixture-of-experts block: router -> top-k experts -> weighted combine.

Written for reading, not speed: a Python loop over experts with gather and index_add.
Production MoE kernels group tokens per expert into one batched matmul, but compute the same thing.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .mlp import SwiGLU


class MoE(nn.Module):
    """Replaces the dense SwiGLU. Each token runs through only k of the E experts.

    forward: x (B, T, D) -> (out (B, T, D), aux_loss scalar)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(hidden_size, intermediate_size) for _ in range(num_experts))
        # Filled on every forward so the tutorial (and tests) can look at the routing.
        self.last_tokens_per_expert: torch.Tensor | None = None
        self.last_dropped: int = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        flat = x.reshape(-1, d)  # (N, D) with N = B*T
        n = flat.shape[0]

        logits = self.router(flat)  # (N, E): one score per expert per token
        probs = torch.softmax(logits.float(), dim=-1)  # (N, E)
        top_p, top_idx = probs.topk(self.top_k, dim=-1)  # (N, k), (N, k)
        top_p = (top_p / top_p.sum(dim=-1, keepdim=True)).to(x.dtype)  # renormalise over the chosen k

        capacity = n * self.top_k  # every assignment fits
        if self.capacity_factor is not None:
            # Each expert gets an equal share of the k*N assignments, times the slack factor.
            capacity = math.ceil(self.capacity_factor * n * self.top_k / self.num_experts)

        out = torch.zeros_like(flat)
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        dropped = 0
        for e, expert in enumerate(self.experts):
            token_ids, slot_ids = torch.nonzero(top_idx == e, as_tuple=True)  # assignments to expert e
            if token_ids.numel() > capacity:
                # Over capacity: earlier tokens win, the rest skip this expert (the residual still carries them).
                dropped += token_ids.numel() - capacity
                token_ids, slot_ids = token_ids[:capacity], slot_ids[:capacity]
            counts[e] = token_ids.numel()
            if token_ids.numel() == 0:
                continue
            y = expert(flat[token_ids]) * top_p[token_ids, slot_ids, None]  # (n_e, D)
            out.index_add_(0, token_ids, y)

        self.last_tokens_per_expert, self.last_dropped = counts, dropped
        return out.view(b, t, d), load_balancing_loss(probs, top_idx, self.num_experts)


def load_balancing_loss(probs: torch.Tensor, top_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch-Transformer auxiliary loss: E * sum_e f_e * P_e.

    f_e: fraction of the N*k assignments that went to expert e (not differentiable).
    P_e: mean router probability for expert e (differentiable, so the gradient flows here).
    It equals 1.0 for perfectly uniform routing and grows as routing collapses onto few experts.
    """
    one_hot = torch.zeros_like(probs).scatter_(1, top_idx, 1.0)  # (N, E)
    f = one_hot.mean(dim=0) / top_idx.shape[1]  # (E,), sums to 1
    p = probs.mean(dim=0)  # (E,), sums to 1
    return num_experts * (f * p).sum()
