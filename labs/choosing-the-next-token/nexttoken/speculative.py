"""Speculative sampling (Leviathan et al., 2022; Chen et al., 2023): draft with q, verify with p, correct on rejection.

    accept a drafted x with probability min(1, p(x) / q(x))
    on rejection, draw from p'(x) proportional to max(0, p(x) - q(x))

The emitted token is then distributed exactly as p. Greedy decoding is the special case where p and q are one-hot
at their argmax: a draft is accepted only if it equals the target's argmax.
"""

from __future__ import annotations

import torch

from .dists import softmax
from .lm import decode


def accept_or_correct(p: torch.Tensor, q: torch.Tensor, x: torch.Tensor, generator=None):
    """One verification. p, q: (B, V); x: (B,) drafted ids. Returns (accepted (B,), token (B,))."""
    px, qx = p.gather(-1, x[:, None])[:, 0], q.gather(-1, x[:, None])[:, 0]
    u = torch.rand(x.shape, generator=generator)
    accepted = u < torch.clamp(px / qx, max=1.0)
    residual = torch.clamp(p - q, min=0)
    residual = residual / residual.sum(-1, keepdim=True).clamp_min(1e-30)
    resample = torch.multinomial(torch.where(residual.sum(-1, keepdim=True) > 0, residual, p), 1,
                                 generator=generator)[:, 0]
    return accepted, torch.where(accepted, x, resample)


def toy_speculative_draws(p: torch.Tensor, q: torch.Tensor, n: int, generator=None) -> torch.Tensor:
    """n independent single-token speculative draws for one target p (V,) and draft q (V,)."""
    P, Q = p.expand(n, -1), q.expand(n, -1)
    x = torch.multinomial(Q, 1, replacement=True, generator=generator)[:, 0]
    return accept_or_correct(P, Q, x, generator)[1]


def _dist(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return softmax(logits, temperature)


@torch.no_grad()
def speculative_round(draft, target, prefix: torch.Tensor, gamma: int, temperature: float = 1.0, generator=None):
    """One draft-and-verify round for B rows that share a prefix length.

    Returns (emitted, n_accepted): emitted is a list of B token lists (1 to gamma + 1 tokens each).
    """
    B = prefix.shape[0]
    q_probs, drafted = [], []

    def pick(logits, _history):
        q = _dist(logits, temperature)
        q_probs.append(q)
        tok = torch.multinomial(q, 1, generator=generator)[:, 0]
        drafted.append(tok)
        return tok

    decode(draft, prefix, pick, gamma)
    xs, qs = torch.stack(drafted, 1), torch.stack(q_probs, 1)  # (B, gamma), (B, gamma, V)
    full = torch.cat([prefix, xs], dim=1)
    t_logits = target(input_ids=full, use_cache=False, logits_to_keep=gamma + 1).logits.float()  # (B, gamma+1, V)
    ps = _dist(t_logits, temperature)  # (B, gamma + 1, V)

    emitted = [[] for _ in range(B)]
    alive = torch.ones(B, dtype=torch.bool)
    n_acc = torch.zeros(B, dtype=torch.long)
    for i in range(gamma):
        acc, tok = accept_or_correct(ps[:, i], qs[:, i], xs[:, i], generator)
        for b in torch.nonzero(alive).flatten().tolist():
            emitted[b].append(int(tok[b]))
        n_acc += (acc & alive).long()
        alive &= acc
    if alive.any():  # every draft accepted: the target's extra position gives one more token for free
        bonus = torch.multinomial(ps[:, gamma], 1, generator=generator)[:, 0]
        for b in torch.nonzero(alive).flatten().tolist():
            emitted[b].append(int(bonus[b]))
    return emitted, n_acc


@torch.no_grad()
def speculative_generate(draft, target, prompt_ids: torch.Tensor, max_new_tokens: int, gamma: int = 4,
                         temperature: float = 1.0, generator=None):
    """Generate one sequence. Returns (tokens, stats). The target is called once per round."""
    ids = prompt_ids
    out, rounds, accepted, drafted = [], 0, 0, 0
    while len(out) < max_new_tokens:
        emitted, n_acc = speculative_round(draft, target, ids, gamma, temperature, generator)
        toks = emitted[0][: max_new_tokens - len(out)]
        out += toks
        ids = torch.cat([ids, torch.tensor([toks])], dim=1)
        rounds += 1
        accepted += int(n_acc[0])
        drafted += gamma
    stats = {"target_calls": rounds, "tokens": len(out), "tokens_per_target_call": len(out) / rounds,
             "draft_acceptance": accepted / drafted}
    return out, stats
