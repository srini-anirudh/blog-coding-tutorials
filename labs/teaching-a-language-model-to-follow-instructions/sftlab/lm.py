"""Loading models, per-token losses, and batched greedy generation."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "HuggingFaceTB/SmolLM2-135M"
BASE_360 = "HuggingFaceTB/SmolLM2-360M"
INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
LLAMA3_TOKENIZER = "NousResearch/Meta-Llama-3-8B-Instruct"  # an ungated copy; only the tokenizer is used


def device() -> str:
    return os.environ.get("SFTLAB_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def chat_tokenizer(repo: str = INSTRUCT):
    """The Instruct tokenizer carries the ChatML template. The base and Instruct SmolLM2 models share a vocabulary."""
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "right"
    return tok


def load_model(repo: str = BASE, dtype=torch.float32, dev: str | None = None):
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype)
    return m.to(dev or device())


def token_nll(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """(B, L, V), (B, L) -> (B, L-1): NLL of token t+1 given tokens up to t, in fp32."""
    return F.cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:], reduction="none")


@torch.no_grad()
def greedy(model, tok, prompts: list[list[int]], max_new: int = 128, stop_ids=None, batch: int = 32,
           sample: bool = False, temperature: float = 1.0, seed: int | None = None) -> list[dict]:
    """Generate from token-id prompts. Returns, per prompt, the new ids (cut after the first stop token),
    whether a stop token was emitted, and the decoded text. Left padding, so any batch size is exact up to
    numerical noise."""
    stop_ids = stop_ids if stop_ids is not None else [tok.eos_token_id]
    dev = next(model.parameters()).device
    out = []
    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(seed)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    res = [None] * len(prompts)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        L = max(len(prompts[i]) for i in idx)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            p = prompts[i]
            ids[r, L - len(p):] = torch.tensor(p)
            att[r, L - len(p):] = 1
        kw = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
        if sample and seed is not None:
            torch.manual_seed(seed + b)
        g = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new,
                           eos_token_id=stop_ids, pad_token_id=pad, **kw)[:, L:].cpu().tolist()
        for r, i in enumerate(idx):
            new, stopped = [], False
            for t in g[r]:
                new.append(t)
                if t in stop_ids:
                    stopped = True
                    break
            res[i] = {"ids": new, "stopped": stopped, "text": tok.decode(new, skip_special_tokens=False)}
    return res
