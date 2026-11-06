"""Every truncation sampler is the same algorithm: build a candidate set C, renormalize over C, sample.

Each rule below takes probabilities (B, V) and returns a boolean keep-mask (B, V). Only the rule that defines C
differs. Every rule keeps the argmax, so C is never empty.

    top-k    : the k most likely tokens                      (fixed size)
    top-p    : smallest prefix with mass >= p                (Holtzman et al., 2019)
    epsilon  : p_i > eps                                     (Hewitt et al., 2022)
    eta      : p_i > min(eps, sqrt(eps) * exp(-H))           (Hewitt et al., 2022)
    min-p    : p_i >= base * max_j p_j                       (Nguyen et al., 2024)
    typical  : smallest set with mass >= tau, ordered by |-log p_i - H|   (Meister et al., 2022)
"""

from __future__ import annotations

import torch

from .dists import entropy, softmax


def _keep_argmax(mask: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    return mask.scatter(-1, probs.argmax(-1, keepdim=True), True)


def top_k_mask(probs: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), probs.shape[-1])
    idx = torch.topk(probs, k, dim=-1).indices
    return torch.zeros_like(probs, dtype=torch.bool).scatter(-1, idx, True)


def top_p_mask(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_p, order = torch.sort(probs, dim=-1, descending=True)
    mass_before = sorted_p.cumsum(-1) - sorted_p  # mass of the tokens ranked above this one
    keep_sorted = mass_before < p
    return torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)


def epsilon_mask(probs: torch.Tensor, eps: float) -> torch.Tensor:
    return _keep_argmax(probs > eps, probs)


def eta_threshold(probs: torch.Tensor, eps: float) -> torch.Tensor:
    """eta = min(eps, sqrt(eps) * exp(-H)), H in nats. Shape (B, 1)."""
    h = entropy(probs).unsqueeze(-1)
    return torch.minimum(torch.full_like(h, eps), eps**0.5 * torch.exp(-h))


def eta_mask(probs: torch.Tensor, eps: float) -> torch.Tensor:
    return _keep_argmax(probs > eta_threshold(probs, eps), probs)


def min_p_mask(probs: torch.Tensor, base: float) -> torch.Tensor:
    return probs >= base * probs.max(dim=-1, keepdim=True).values


def typical_mask(probs: torch.Tensor, tau: float) -> torch.Tensor:
    h = entropy(probs).unsqueeze(-1)
    surprisal = torch.where(probs > 0, -probs.log(), torch.full_like(probs, float("inf")))
    deviation = (surprisal - h).abs()
    order = torch.argsort(deviation, dim=-1)  # most typical first
    sorted_p = probs.gather(-1, order)
    keep_sorted = (sorted_p.cumsum(-1) - sorted_p) < tau
    return torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)


def full_mask(probs: torch.Tensor, _param=None) -> torch.Tensor:
    """Plain ancestral sampling: C = V."""
    return torch.ones_like(probs, dtype=torch.bool)


RULES = {
    "full": full_mask,
    "top_k": top_k_mask,
    "top_p": top_p_mask,
    "epsilon": epsilon_mask,
    "eta": eta_mask,
    "min_p": min_p_mask,
    "typical": typical_mask,
}


def candidate_mask(logits: torch.Tensor, rule: str, param, temperature: float = 1.0,
                   order: str = "truncate_then_temper") -> torch.Tensor:
    """Keep-mask for `rule`. `order` says whether C is computed from the model's own p or from the tempered p."""
    if order == "truncate_then_temper":
        probs = softmax(logits)
    elif order == "temper_then_truncate":
        probs = softmax(logits, temperature)
    else:
        raise ValueError(order)
    return RULES[rule](probs, param)


def truncated_probs(logits: torch.Tensor, rule: str, param, temperature: float = 1.0,
                    order: str = "truncate_then_temper") -> torch.Tensor:
    """Steps 1 and 2: the candidate set, then temperature and renormalization inside it."""
    mask = candidate_mask(logits, rule, param, temperature, order)
    z = (logits.float() / temperature).masked_fill(~mask, float("-inf"))
    return torch.softmax(z, dim=-1)


def sample(logits: torch.Tensor, rule: str = "full", param=None, temperature: float = 1.0,
           order: str = "truncate_then_temper", generator: torch.Generator | None = None) -> torch.Tensor:
    """Step 3. logits (B, V) -> token ids (B,). temperature 0 means argmax."""
    if temperature == 0:
        return logits.argmax(-1)
    probs = truncated_probs(logits, rule, param, temperature, order)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
