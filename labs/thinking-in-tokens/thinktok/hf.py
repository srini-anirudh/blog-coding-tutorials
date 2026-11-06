"""A `generate` callable for thinktok.budget, backed by a Hugging Face causal LM. Used by the GPU scripts."""

from __future__ import annotations

import torch


def load(model_id: str = "Qwen/Qwen3-0.6B", device: str | None = None, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, tok


def make_generate(model, tok, temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20,
                  batch_size: int = 64, seed: int = 0):
    """Sampling defaults are the Qwen3 model card's for thinking mode. temperature 0 means greedy."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    calls = {"n": 0}

    @torch.no_grad()
    def generate(prompts: list[list[int]], max_new_tokens: int, stop_ids: set[int]) -> list[list[int]]:
        if max_new_tokens <= 0:
            return [[] for _ in prompts]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # less padding per batch
        out: list[list[int] | None] = [None] * len(prompts)
        for s in range(0, len(order), batch_size):
            chunk = order[s : s + batch_size]
            width = max(len(prompts[i]) for i in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, i in enumerate(chunk):
                p = prompts[i]
                ids[r, width - len(p):] = torch.tensor(p)
                mask[r, width - len(p):] = 1
            torch.manual_seed(seed + 7919 * calls["n"] + s)
            sample = temperature > 0
            gen = model.generate(
                input_ids=ids.to(model.device), attention_mask=mask.to(model.device), max_new_tokens=max_new_tokens,
                do_sample=sample, temperature=temperature if sample else None, top_p=top_p if sample else None,
                top_k=top_k if sample else None, eos_token_id=sorted(stop_ids), pad_token_id=pad,
            )[:, width:].tolist()
            for r, i in enumerate(chunk):
                new = gen[r]
                for j, t in enumerate(new):
                    if t in stop_ids:
                        new = new[: j + 1]
                        break
                else:
                    new = [t for t in new if t != pad] if pad not in stop_ids else new
                out[i] = new
        calls["n"] += 1
        return out  # type: ignore[return-value]

    return generate
