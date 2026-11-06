"""Per-token and per-sequence log-probabilities: the one quantity DPO, REINFORCE and GRPO all need.

    log pi(y | x) = sum_t log pi(y_t | x, y_<t)

Logits at position t predict the token at position t + 1, so every function here shifts by one.
"""

from __future__ import annotations

import torch


def gather_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log softmax(logits)[target] without materialising the full (B, T, V) log-softmax.

    logits: (B, T, V)   targets: (B, T)   ->   (B, T)
    """
    logits = logits.float()  # bf16 keeps about 3 significant digits; ratios of log-probs need more
    picked = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim=-1)


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions that skip left padding, matching what generate() used: (B, L) -> (B, L)."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def per_token_logprobs(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int | None = None
) -> torch.Tensor:
    """Log-prob of every token given its prefix.

    input_ids, attention_mask: (B, L).
    Returns (B, L - 1) aligned with input_ids[:, 1:], or (B, K) for the last K tokens if logits_to_keep=K.
    """
    kwargs = {}
    if logits_to_keep is not None:
        # One extra position: the logit before the first kept token predicts it.
        kwargs["logits_to_keep"] = logits_to_keep + 1
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        use_cache=False,
        **kwargs,
    )
    logits = out.logits[:, :-1]  # (B, L-1, V) or (B, K, V)
    targets = input_ids[:, 1:] if logits_to_keep is None else input_ids[:, -logits_to_keep:]
    return gather_logprobs(logits, targets)


def sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    average: bool = False,
) -> torch.Tensor:
    """Sum (or mean) of log-probs over completion tokens only.

    input_ids, attention_mask, completion_mask: (B, L), completion_mask = 1 on response tokens.
    Returns (B,).
    """
    logps = per_token_logprobs(model, input_ids, attention_mask)  # (B, L-1)
    mask = completion_mask[:, 1:].to(logps.dtype)  # token t+1 is scored by logits at t
    total = (logps * mask).sum(-1)
    return total / mask.sum(-1).clamp(min=1) if average else total
