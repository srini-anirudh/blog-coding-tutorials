"""Greedy decoding written out by hand, to see what `lm.generate(inputs_embeds=...)` does for us.

Prefill: run the whole multimodal prompt (B x N x d_lm embeddings) once and keep the KV cache.
Decode:  feed only the newest token id; images are never re-encoded, their keys and values are cached.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from .modeling import TinyVLM
from .positions import position_ids_1d


@torch.no_grad()
def greedy_generate(
    model: TinyVLM,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor | None,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 20,
    eos_token_id: int | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Returns B x T_new new token ids. use_cache=False recomputes the full sequence every step (slow reference)."""
    embeds = model.embed(input_ids, pixel_values)  # B x N x d_lm, image features already inside
    mask = attention_mask.clone()
    cache = DynamicCache(config=model.lm.config) if use_cache else None
    new_tokens = []
    step_input, positions = embeds, position_ids_1d(mask)
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    for _ in range(max_new_tokens):
        out = model.lm(
            inputs_embeds=step_input, attention_mask=mask, position_ids=positions, past_key_values=cache, use_cache=use_cache
        )
        next_token = out.logits[:, -1].argmax(-1)  # B
        if eos_token_id is not None:
            next_token = next_token.masked_fill(finished, eos_token_id)
            finished |= next_token == eos_token_id
        new_tokens.append(next_token)
        next_embed = model.get_input_embeddings()(next_token)[:, None]  # B x 1 x d_lm
        mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
        if use_cache:
            step_input, positions = next_embed, position_ids_1d(mask)[:, -1:]
        else:
            embeds = torch.cat([embeds, next_embed], dim=1)
            step_input, positions = embeds, position_ids_1d(mask)
        if eos_token_id is not None and finished.all():
            break
    return torch.stack(new_tokens, dim=1)
