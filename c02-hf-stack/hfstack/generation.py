"""What `model.generate()` does, written out as a plain loop.

    generate()
     ├── prefill: one forward pass over the prompt, filling the KV cache
     ├── decode: forward pass on the newest token only, reading the cache
     ├── logits processors: temperature, top-k, top-p reshape the last-position logits
     ├── sampler: argmax (greedy) or a draw from softmax
     └── stopping criteria: EOS or max_new_tokens

`tests/test_generation.py` checks that these loops produce the same tokens as `model.generate`.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache


def positions_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position IDs that skip left padding. attention_mask: (B, T) of 0/1 → (B, T) positions."""
    return (attention_mask.cumsum(-1) - 1).clamp(min=0)


@torch.no_grad()
def greedy_no_cache(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Reference loop: re-run the whole sequence every step. Cost per step grows with length. (B, T) → (B, T+K)."""
    for _ in range(max_new_tokens):
        logits = model(input_ids=input_ids, use_cache=False).logits  # (B, T, V)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids


def process_logits(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0
) -> torch.Tensor:
    """Apply temperature, then top-k, then top-p to last-position logits (B, V). Removed entries become -inf."""
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]  # (B, 1)
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        # Drop a token if the tokens ranked above it already cover top_p of the mass. Keep at least one.
        drop_sorted = (cumulative - sorted_logits.softmax(-1)) >= top_p
        drop_sorted[..., 0] = False
        drop = drop_sorted.scatter(-1, order, drop_sorted)
        logits = logits.masked_fill(drop, float("-inf"))
    return logits


@torch.no_grad()
def generate_with_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 20,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Prefill once, then decode one token per step from the KV cache.

    input_ids: (B, T) left-padded prompts. Returns (B, T + K) with K ≤ max_new_tokens.
    Finished rows are filled with `pad_token_id`, as `generate` does.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    cache = DynamicCache(config=model.config)
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    step_input = input_ids  # prefill: the whole prompt

    for _ in range(max_new_tokens):
        position_ids = positions_from_mask(attention_mask)[:, -step_input.size(1) :]
        out = model(
            input_ids=step_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        logits = out.logits[:, -1, :].float()  # (B, V): only the last position predicts the next token
        if do_sample:
            probs = process_logits(logits, temperature, top_k, top_p).softmax(-1)
            next_token = torch.multinomial(probs, 1, generator=generator).squeeze(-1)  # (B,)
        else:
            next_token = logits.argmax(-1)
        next_token = torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)

        input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token[:, None])], dim=-1)
        step_input = next_token[:, None]  # decode: only the new token goes through the model

        if eos_token_id is not None:
            finished |= next_token == eos_token_id
            if finished.all():
                break
    return input_ids
