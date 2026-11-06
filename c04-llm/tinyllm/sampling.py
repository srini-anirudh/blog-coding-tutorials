"""Turn one row of logits into one token: greedy, temperature, top-k, top-p."""

from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by T. T < 1 sharpens the distribution, T > 1 flattens it."""
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits per row, set the rest to -inf. logits: (B, V)."""
    k = min(k, logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]  # (B, 1): the k-th largest value
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set of tokens whose probability sums to at least p.

    logits: (B, V). The most likely token is always kept, so the set is never empty.
    """
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    # Probability mass strictly before each token. A token is dropped once that mass already reaches p.
    mass_before = torch.cumsum(probs, dim=-1) - probs
    sorted_logits = sorted_logits.masked_fill(mass_before >= p, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: (B, V) for the last position -> next token ids (B, 1).

    temperature == 0 means greedy decoding. Otherwise the filters run in the order Hugging Face
    uses: temperature, then top-k, then top-p, then a draw from the renormalised distribution.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = apply_temperature(logits.float(), temperature)
    if top_k is not None:
        logits = apply_top_k(logits, top_k)
    if top_p is not None:
        logits = apply_top_p(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
