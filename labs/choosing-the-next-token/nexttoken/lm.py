"""Small helpers around a Hugging Face causal LM: load it, run a decode loop with a KV cache, score text.

The decode loop is deliberately bare: logits -> optional processors -> a picker that returns token ids. Every decoder
in this lab plugs into it as a `pick(logits, history)` function.
"""

from __future__ import annotations

import os
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

BASE = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"
LARGE = "HuggingFaceTB/SmolLM2-360M"  # same tokenizer as BASE
SCORER = "Qwen/Qwen2.5-0.5B"  # a different model family, used only to score text


def load(name: str, device: str | None = None):
    """Load in float32. The device defaults to $NEXTTOKEN_DEVICE, else CPU. Decoding logic always runs on CPU:
    the helpers below move ids to the model's device and bring logits back."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    return tok, model.to(device or os.environ.get("NEXTTOKEN_DEVICE", "cpu"))


@torch.no_grad()
def next_logits(model, ids: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, V) logits at the last position, no cache."""
    return model(input_ids=ids.to(model.device), use_cache=False, logits_to_keep=1).logits[:, -1, :].float().cpu()


@torch.no_grad()
def all_logits(model, ids: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, T, V): the next-token logits at every position."""
    return model(input_ids=ids.to(model.device), use_cache=False).logits.float().cpu()


Picker = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (logits (B, V), history (B, t)) -> ids (B,)


@torch.no_grad()
def decode(model, prompt_ids: torch.Tensor, pick: Picker, max_new_tokens: int,
           eos_id: int | None = None, return_logits: bool = False):
    """Prefill once, then one cached forward per token. prompt_ids (B, T), all rows the same prompt length.

    Returns generated ids (B, K). With return_logits, also the raw model logits (B, K, V) used at each step,
    before the picker touched them, so callers can measure what the model itself believed.
    """
    cache = DynamicCache(config=model.config)
    step = prompt_ids.to(model.device)
    history = prompt_ids
    out, raw = [], []
    done = torch.zeros(prompt_ids.shape[0], dtype=torch.bool)
    for _ in range(max_new_tokens):
        logits = model(input_ids=step, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1, :].float().cpu()
        nxt = pick(logits, history)
        if eos_id is not None:
            nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
            done |= nxt == eos_id
        out.append(nxt)
        if return_logits:
            raw.append(logits)
        history = torch.cat([history, nxt.unsqueeze(-1)], dim=-1)
        step = nxt.unsqueeze(-1).to(model.device)
        if eos_id is not None and done.all():
            break
    gen = torch.stack(out, dim=1)
    return (gen, torch.stack(raw, dim=1)) if return_logits else gen


@torch.no_grad()
def token_logprobs(model, prompt_ids: torch.Tensor, cont_ids: torch.Tensor) -> torch.Tensor:
    """log p(cont_t | prompt, cont_<t) for one prompt (1, T) and one continuation (1, K). Returns (K,)."""
    ids = torch.cat([prompt_ids, cont_ids], dim=-1)
    logp = torch.log_softmax(all_logits(model, ids)[0, prompt_ids.shape[1] - 1:-1], dim=-1)
    return logp.gather(-1, cont_ids[0].unsqueeze(-1)).squeeze(-1)


def text_nll(tok, model, prompt: str, continuation: str) -> float:
    """Mean negative log-likelihood per token of `continuation` given `prompt`, in the scorer's own tokenization."""
    p = tok(prompt, return_tensors="pt").input_ids
    c = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    if c.shape[1] == 0:
        return float("nan")
    return -token_logprobs(model, p, c).mean().item()
