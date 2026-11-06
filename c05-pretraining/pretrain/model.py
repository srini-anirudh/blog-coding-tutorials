"""The model we pretrain.

Default: Hugging Face `LlamaForCausalLM` built from a small `LlamaConfig`. It is the
same architecture C4 builds by hand (RMSNorm, RoPE, GQA attention, SwiGLU MLP), so this
folder runs on its own. With `model_type="tinyllm"` we import C4's `TinyLLM` from
`../c04-llm` instead. The trainer only needs `model(input_ids).logits`.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch
import torch.nn as nn

from pretrain.config import ModelArgs

C4_DIR = Path(__file__).resolve().parents[2] / "c04-llm"


def build_llama(a: ModelArgs) -> nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=a.vocab_size,
        hidden_size=a.hidden_size,
        intermediate_size=a.intermediate_size,
        num_hidden_layers=a.num_layers,
        num_attention_heads=a.num_heads,
        num_key_value_heads=a.num_kv_heads,
        max_position_embeddings=a.max_seq_len,
        rope_theta=a.rope_theta,
        tie_word_embeddings=a.tie_embeddings,
        attn_implementation="sdpa",
    )
    return LlamaForCausalLM(cfg)


def build_tinyllm(a: ModelArgs) -> nn.Module:
    """C4's TinyLLM. Fields that C4's ModelConfig does not define are skipped."""
    if not C4_DIR.exists():
        raise FileNotFoundError(f"--model tinyllm needs C4 at {C4_DIR}")
    sys.path.insert(0, str(C4_DIR))
    from tinyllm import ModelConfig, TinyLLM  # noqa: E402  (C4's public contract)

    known = {f.name for f in dataclasses.fields(ModelConfig)}
    values = dataclasses.asdict(a)
    aliases = {"tie_embeddings": ["tie_embeddings", "tie_word_embeddings"], "rope_theta": ["rope_theta", "rope_base"]}
    kwargs = {}
    for key, value in values.items():
        for name in aliases.get(key, [key]):
            if name in known:
                kwargs[name] = value
    return TinyLLM(ModelConfig(**kwargs))


def build_model(a: ModelArgs, model_type: str = "llama") -> nn.Module:
    if model_type == "llama":
        return build_llama(a)
    if model_type == "tinyllm":
        return build_tinyllm(a)
    raise ValueError(f"unknown model_type {model_type!r}")


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The list of transformer blocks: the largest ModuleList in the model."""
    lists = [m for m in model.modules() if isinstance(m, nn.ModuleList)]
    if not lists:
        raise ValueError("model has no ModuleList of blocks")
    return max(lists, key=len)


def count_params(model: nn.Module, non_embedding: bool = False) -> int:
    """Unique parameters (tied weights counted once). non_embedding drops token embeddings."""
    vocab = getattr(getattr(model, "config", None), "vocab_size", -1)
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if non_embedding and p.dim() == 2 and p.shape[0] == vocab:
            continue
        total += p.numel()
    return total


@torch.no_grad()
def sample(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 0.8,
    top_k: int = 40,
    max_seq_len: int = 256,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Plain autoregressive sampling, no KV cache (C4 covers the cache).

    Stops at EOS: the model learned to emit EOS because every training document ended in one.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], device=device)
    for _ in range(max_new_tokens):
        logits = model(ids[:, -max_seq_len:]).logits[0, -1].float()  # (V,)
        logits = logits / temperature
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = -float("inf")
        probs = torch.softmax(logits, -1).cpu()
        nxt = torch.multinomial(probs, 1, generator=generator).item()
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], 1)
        if nxt == eos_id:
            break
    return ids[0, len(prompt_ids):].tolist()
