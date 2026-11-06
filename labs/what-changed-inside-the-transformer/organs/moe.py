"""A top-k MoE layer with three ways to keep the load balanced: none, the Switch auxiliary loss, and DeepSeek-V3's
auxiliary-loss-free bias. Experts are C4's SwiGLU; the auxiliary loss is C4's load_balancing_loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import _paths  # noqa: F401
from tinyllm.mlp import SwiGLU
from tinyllm.moe import load_balancing_loss


class BalancedMoE(nn.Module):
    """x (N, d) -> y (N, d), plus the auxiliary loss (zero unless balance == "aux").

    balance: "none" | "aux" | "bias". With "bias", a per-expert bias is added to the router scores for the top-k
    choice only. It never enters the gate weights or the loss; `update_bias` nudges it after each step.
    """

    def __init__(self, d: int, width: int, experts: int, top_k: int, balance: str = "none", shared: int = 0,
                 head_start: float = 0.0) -> None:
        super().__init__()
        self.top_k, self.balance, self.n = top_k, balance, experts
        self.router = nn.Linear(d, experts, bias=True)
        nn.init.zeros_(self.router.bias)
        with torch.no_grad():  # "expert 0 is slightly better early": a learnable head start on its score
            self.router.bias[0] = head_start
        self.experts = nn.ModuleList(SwiGLU(d, width) for _ in range(experts))
        self.shared = nn.ModuleList(SwiGLU(d, width) for _ in range(shared))
        self.register_buffer("bias", torch.zeros(experts))
        self.last_load = torch.zeros(experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        probs = logits.float().softmax(-1)
        choice_scores = logits + self.bias if self.balance == "bias" else logits
        top_idx = choice_scores.topk(self.top_k, -1).indices  # (N, k)
        gate = probs.gather(1, top_idx)
        gate = gate / gate.sum(-1, keepdim=True)
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            rows, slots = torch.nonzero(top_idx == e, as_tuple=True)
            if rows.numel():
                y.index_add_(0, rows, expert(x[rows]) * gate[rows, slots, None])
        for expert in self.shared:  # every token, no routing
            y = y + expert(x)
        self.last_load = torch.bincount(top_idx.flatten(), minlength=self.n).float() / top_idx.numel()
        aux = load_balancing_loss(probs, top_idx, self.n) if self.balance == "aux" else x.new_zeros(())
        return y, aux

    @torch.no_grad()
    def update_bias(self, gamma: float) -> None:
        """Raise the bias of under-loaded experts, lower it for over-loaded ones (sign rule, DeepSeek-V3)."""
        if self.balance == "bias":
            self.bias += gamma * torch.sign(1.0 / self.n - self.last_load)


def clustered_task(n: int, d: int, clusters: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens from `clusters` Gaussian blobs; each blob has its own linear map to the target, so experts can specialize."""
    g = torch.Generator().manual_seed(1000)  # the task is fixed; `seed` only picks which samples are drawn
    centers = torch.randn(clusters, d, generator=g) * 3
    maps = torch.randn(clusters, d, d, generator=g) / d**0.5
    gs = torch.Generator().manual_seed(seed)
    c = torch.randint(0, clusters, (n,), generator=gs)
    x = centers[c] + torch.randn(n, d, generator=gs)
    y = torch.tanh(torch.einsum("nij,nj->ni", maps[c], x))
    return x, y


def train_moe(balance: str, seed: int, steps: int = 600, experts: int = 16, top_k: int = 2, d: int = 32,
              aux_coef: float = 0.01, gamma: float = 0.01, head_start: float = 0.0) -> dict[str, float]:
    torch.manual_seed(seed)
    moe = BalancedMoE(d, 32, experts, top_k, balance, head_start=head_start)
    xtr, ytr = clustered_task(16384, d, 32, seed)
    xte, yte = clustered_task(4096, d, 32, seed + 100)
    opt = torch.optim.Adam(moe.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, xtr.shape[0], (512,), generator=g)
        y, aux = moe(xtr[idx])
        loss = F.mse_loss(y, ytr[idx]) + aux_coef * aux
        opt.zero_grad()
        loss.backward()
        opt.step()
        moe.update_bias(gamma)
    with torch.no_grad():
        y, _ = moe(xte)
        out = {"test_mse": F.mse_loss(y, yte).item()}
    load = moe.last_load
    entropy = -(load.clamp_min(1e-12) * load.clamp_min(1e-12).log()).sum().item()
    out.update(max_over_mean=load.max().item() * experts, min_over_mean=load.min().item() * experts,
               dead=int((load < 0.1 / experts).sum().item()),
               entropy_frac=entropy / torch.log(torch.tensor(float(experts))).item())
    return out
