"""Batched sampling with left padding, and the completion mask every RL loss needs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from .logprobs import per_token_logprobs


@dataclass
class Generations:
    prompt_ids: torch.Tensor  # (N, P) left-padded
    prompt_mask: torch.Tensor  # (N, P)
    completion_ids: torch.Tensor  # (N, C) right-padded
    completion_mask: torch.Tensor  # (N, C) 1 up to and including the first <|im_end|>
    texts: list[str]
    truncated: torch.Tensor  # (N,) bool, no <|im_end|> within max_new_tokens


def left_pad(prompts: list[list[int]], pad_id: int, multiple: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Left padding keeps the last prompt token of every row in the same column, which generate() needs.

    The width is rounded up to `multiple` so MPS sees few distinct shapes.
    """
    P = -(-max(len(p) for p in prompts) // multiple) * multiple
    ids = torch.full((len(prompts), P), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), P), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, P - len(p) :] = torch.tensor(p)
        mask[i, P - len(p) :] = 1
    return ids, mask


def completion_mask_from_ids(completion_ids: torch.Tensor, eos_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """1 for tokens up to and including the first eos, 0 after it. (N, C) -> (N, C), (N,)."""
    is_eos = completion_ids == eos_id
    has_eos = is_eos.any(-1)
    first = torch.where(has_eos, is_eos.int().argmax(-1), completion_ids.shape[1] - 1)
    cols = torch.arange(completion_ids.shape[1], device=completion_ids.device)
    return (cols[None, :] <= first[:, None]).long(), ~has_eos


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    prompts: list[list[int]],
    max_new_tokens: int = 48,
    temperature: float = 1.0,
    do_sample: bool = True,
) -> Generations:
    device = next(model.parameters()).device
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    ids, mask = left_pad(prompts, tok.pad_token_id)
    ids, mask = ids.to(device), mask.to(device)
    # Sample from the full distribution, so the log-probs we score later are the ones that were sampled.
    sampling = {"temperature": temperature, "top_p": 1.0, "top_k": 0} if do_sample else {}
    out = model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=eos_id,
        pad_token_id=tok.pad_token_id,
        **sampling,
    )
    completion = out[:, ids.shape[1] :]
    cmask, truncated = completion_mask_from_ids(completion, eos_id)
    completion = completion.masked_fill(cmask == 0, tok.pad_token_id)
    texts = tok.batch_decode(completion, skip_special_tokens=True)
    return Generations(ids, mask, completion, cmask, texts, truncated)


def score_completions(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token log-probs of the completion under `model`: (N, C). Same layout as TRL's GRPOTrainer."""
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (N, P + C)
    attn = torch.cat([prompt_mask, completion_mask], dim=1)
    return per_token_logprobs(model, input_ids, attn, logits_to_keep=completion_ids.shape[1])
