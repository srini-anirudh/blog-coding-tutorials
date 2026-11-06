"""Why three projections: what goes wrong when the scores use the raw vectors, or one shared map.

Three score rules, all followed by the same learned value map:
  none      s_ij = x_i . x_j              symmetric, and x_i . x_i is large
  tied      s_ij = (W x_i) . (W x_j)      still symmetric: M = W W^T
  separate  s_ij = (W_Q x_i) . (W_K x_j)  M = W_Q W_K^T, any matrix of rank <= d_k

Two directional probe tasks:
  cycle3    three tokens of types A, B, C, each with a payload. A must read B's payload, B must read C's,
            C must read A's. With symmetric scores, A prefers B means s_AB > s_AC; B prefers C means
            s_BC > s_BA = s_AB; C prefers A means s_CA > s_CB = s_BC. So s_AC < s_AB < s_BC < s_CA = s_AC,
            a contradiction. At least one of the three rows must fail.
  previous  position i copies the token at i - 1, cyclic over n positions. The same chain argument rules
            out one row per cycle, so symmetric scores can still reach (n - 1) / n.
Softmax normalises each row on its own, so symmetry of the scores does not make the weights symmetric.
It only forbids directed cycles of preference.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .common import attention


def raw_self_attention_stats(n: int = 64, d: int = 64, scaled: bool = False, seed: int = 0) -> dict:
    """softmax(X X^T) X on random vectors: how symmetric, how self-focused."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    scores = x @ x.T / (math.sqrt(d) if scaled else 1.0)
    w = torch.softmax(scores, dim=-1)
    return {
        "max |S - S^T|": (scores - scores.T).abs().max().item(),
        "rows whose argmax is self": (w.argmax(-1) == torch.arange(n)).float().mean().item(),
        "mean self weight": w.diagonal().mean().item(),
    }


class OneLayerRouter(nn.Module):
    """Embed tokens and positions, run one attention layer, read the answer from its output only."""

    def __init__(self, vocab: int, n: int, d: int = 32, rule: str = "separate", d_k: int | None = None, use_pos: bool = True) -> None:
        super().__init__()
        if rule not in ("none", "tied", "separate"):
            raise ValueError(rule)
        self.rule, self.use_pos, d_k = rule, use_pos, d_k or d
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(n, d)
        self.w_q = nn.Linear(d, d_k, bias=False)
        self.w_k = nn.Linear(d, d_k, bias=False)  # unused unless rule == "separate"
        self.w_v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, vocab)

    def scores_qk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rule == "none":
            return x, x
        if self.rule == "tied":
            return self.w_q(x), self.w_q(x)
        return self.w_q(x), self.w_k(x)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tok(tokens) + (self.pos.weight[: tokens.size(1)] if self.use_pos else 0.0)
        q, k = self.scores_qk(x)
        z, w = attention(q, k, self.w_v(x))
        return self.out(z), w


def previous_token_batch(batch: int, g: torch.Generator, n: int = 12, vocab: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(vocab, (batch, n), generator=g)
    return tokens, tokens.roll(1, dims=1)  # target at i is the token at i - 1, cyclic


def cycle3_batch(batch: int, g: torch.Generator, payloads: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Token id = type * payloads + payload. Types 0, 1, 2 appear once each, in random order.
    The target of a type-t token is the payload of the type-(t + 1) % 3 token."""
    types = torch.argsort(torch.rand(batch, 3, generator=g), dim=1)
    pay = torch.randint(payloads, (batch, 3), generator=g)
    by_type = torch.empty_like(pay).scatter_(1, types, pay)  # payload of each type
    target = by_type.gather(1, (types + 1) % 3)
    return types * payloads + pay, target


TASKS = {  # name: (batch function, sequence length, vocabulary, number of classes, use positions)
    "cycle3": (cycle3_batch, 3, 24, 8, False),
    "previous": (previous_token_batch, 12, 12, 12, True),
}


def train_router(rule: str, task: str = "cycle3", d_k: int | None = None, steps: int = 1500, seed: int = 0) -> dict:
    """Train one router on a directional task. Returns test accuracy and attention statistics."""
    make, n, vocab, classes, use_pos = TASKS[task]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = OneLayerRouter(vocab, n, rule=rule, d_k=d_k, use_pos=use_pos)
    model.out = nn.Linear(model.w_v.out_features, classes)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        tokens, target = make(256, g)
        logits, _ = model(tokens)
        loss = nn.functional.cross_entropy(logits.reshape(-1, classes), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        tokens, target = make(4096, torch.Generator().manual_seed(seed + 999))
        logits, w = model(tokens)
        correct = (logits.argmax(-1) == target).float()
        out = {"accuracy": correct.mean().item(), "final train loss": loss.item()}
        if task == "cycle3":  # accuracy per query type: symmetric rules should give up one type
            types = tokens // (vocab // 3)
            out["accuracy by type"] = [round(correct[types == t].mean().item(), 3) for t in range(3)]
        else:
            idx = torch.arange(n)
            out["weight on i-1"] = w[:, idx, (idx - 1) % n].mean().item()
            out["weight on self"] = w[:, idx, idx].mean().item()
        return out


def identifiability_gap(d: int = 32, d_k: int = 8, seed: int = 0) -> dict:
    """Only W_Q W_K^T matters: (W_Q A, W_K A^-T) gives the same scores, and the rank is at most d_k."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(10, d, generator=g, dtype=torch.float64)
    w_q, w_k = torch.randn(d, d_k, generator=g, dtype=torch.float64), torch.randn(d, d_k, generator=g, dtype=torch.float64)
    a = torch.randn(d_k, d_k, generator=g, dtype=torch.float64) + 3 * torch.eye(d_k, dtype=torch.float64)
    s1 = (x @ w_q) @ (x @ w_k).T
    s2 = (x @ w_q @ a) @ (x @ w_k @ torch.linalg.inv(a).T).T
    return {
        "max score change after reparameterising": (s1 - s2).abs().max().item(),
        "max weight change in W_Q": (w_q @ a - w_q).abs().max().item(),
        "rank of W_Q W_K^T": int(torch.linalg.matrix_rank(w_q @ w_k.T)),
        "d_model": d,
        "d_k": d_k,
    }


def permutation_check(n: int = 10, d: int = 16, seed: int = 0) -> dict:
    """Shuffle the inputs: without positions the outputs shuffle the same way; with positions they do not."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w_q, w_k, w_v = (torch.randn(d, d, generator=g) / math.sqrt(d) for _ in range(3))
    pos = torch.randn(n, d, generator=g)
    perm = torch.randperm(n, generator=g)

    def layer(h):
        return attention(h @ w_q, h @ w_k, h @ w_v)[0]

    return {
        "no positions: max |f(Px) - P f(x)|": (layer(x[perm]) - layer(x)[perm]).abs().max().item(),
        "positions added: max |f(Px + p) - P f(x + p)|": (layer(x[perm] + pos) - layer(x + pos)[perm]).abs().max().item(),
    }
