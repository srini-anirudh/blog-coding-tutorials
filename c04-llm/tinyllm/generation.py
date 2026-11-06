"""Autoregressive generation, with and without a KV cache."""

from __future__ import annotations

import torch

from .sampling import sample_next_token


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    eos_token_id: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Extend input_ids (B, T) by up to max_new_tokens tokens. Returns (B, T + new).

    use_cache=False re-runs the whole sequence every step: simple, and O(T^2) work per token.
    use_cache=True runs the prompt once (prefill), then feeds one token per step (decode).
    Both paths must produce the same tokens; tests/test_cache.py checks that.
    """
    was_training = model.training
    model.eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)

    ids = input_ids
    cache = None
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        if not use_cache:
            logits = model(ids).logits  # (B, T_so_far, V), all recomputed
        elif cache is None:
            out = model(ids, use_cache=True)  # prefill: the whole prompt, fills the cache
            logits, cache = out.logits, out.past_key_values
        else:
            logits = model(ids[:, -1:], past_key_values=cache).logits  # decode: (B, 1, V)

        next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, generator)  # (B, 1)
        if eos_token_id is not None:
            # Rows that already ended keep emitting EOS, so the batch stays rectangular.
            next_token = torch.where(finished[:, None], torch.full_like(next_token, eos_token_id), next_token)
            finished |= next_token[:, 0] == eos_token_id
        ids = torch.cat((ids, next_token), dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break
    model.train(was_training)
    return ids
