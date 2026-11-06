"""Truncation as desmoothing, at toy scale.

We know the true next-token distributions exactly: each context has a limited support (from 1 to many tokens) over a
vocabulary of V tokens. A softmax model with a d-dimensional bottleneck, logits = W h_c, is trained by cross-entropy on
samples from them. Then we ask how much mass it leaks outside the true support, and how well each truncation rule
recovers that support.
"""

from __future__ import annotations

import torch

from .truncation import RULES


def make_true_distributions(n_contexts: int, vocab: int, max_support: int, seed: int = 0):
    """Each context gets a random support size in [1, max_support], spaced log-uniformly, and Zipf-like weights."""
    g = torch.Generator().manual_seed(seed)
    sizes = torch.logspace(0, torch.log10(torch.tensor(float(max_support))), n_contexts).round().long()
    sizes = sizes[torch.randperm(n_contexts, generator=g)]
    p = torch.zeros(n_contexts, vocab)
    for c, s in enumerate(sizes.tolist()):
        idx = torch.randperm(vocab, generator=g)[:s]
        w = 1.0 / torch.arange(1, s + 1).float() ** 0.8
        p[c, idx] = w / w.sum()
    return p


class BottleneckLM(torch.nn.Module):
    def __init__(self, n_contexts: int, vocab: int, d: int):
        super().__init__()
        self.h = torch.nn.Embedding(n_contexts, d)
        self.unembed = torch.nn.Linear(d, vocab, bias=True)

    def forward(self, c):
        return self.unembed(self.h(c))


def train(p_true: torch.Tensor, d: int, samples_per_context: int, steps: int = 3000, lr: float = 0.02,
          seed: int = 0) -> BottleneckLM:
    """Maximum likelihood on a finite sample: draw tokens from p_true, minimize cross-entropy with Adam."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    C, V = p_true.shape
    tokens = torch.multinomial(p_true, samples_per_context, replacement=True, generator=g)  # (C, S)
    # Cross-entropy on the sample equals cross-entropy against the empirical distribution of the sample.
    p_emp = torch.zeros(C, V).scatter_add_(1, tokens, torch.ones_like(tokens, dtype=torch.float)) / samples_per_context
    model = BottleneckLM(C, V, d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ctx = torch.arange(C)
    for _ in range(steps):
        loss = -(p_emp * torch.log_softmax(model(ctx), -1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def support_report(p_model: torch.Tensor, p_true: torch.Tensor, rules: dict[str, object],
                   rows: torch.Tensor | None = None) -> dict:
    """Leaked mass outside the true support, and per-rule support precision and recall averaged over contexts.

    precision: kept tokens that are in the true support / kept tokens
    recall   : true-support tokens that are kept / true-support size
    """
    if rows is not None:  # restrict to a subset of contexts, e.g. only the sharp ones
        p_model, p_true = p_model[rows], p_true[rows]
    support = p_true > 0
    out = {
        "leaked_mass_mean": float((p_model * ~support).sum(-1).mean()),
        "min_model_prob": float(p_model.min()),
        "rules": {},
    }
    for name, param in rules.items():
        keep = RULES[name](p_model, param)
        tp = (keep & support).sum(-1).float()
        out["rules"][name] = {
            "param": param,
            "precision": float((tp / keep.sum(-1)).mean()),
            "recall": float((tp / support.sum(-1)).mean()),
            "mean_set_size": float(keep.sum(-1).float().mean()),
        }
    return out
