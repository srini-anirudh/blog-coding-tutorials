"""Loading the policy, a frozen reference copy, and a tiny random model for fast tests."""

from __future__ import annotations

import copy

import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from .utils import BASE_MODEL


def load_policy(name: str = BASE_MODEL, device: torch.device | str = "cpu", dtype=torch.float32):
    """The pretrained base (or an SFT checkpoint directory), on `device`."""
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    return model.to(device)


def random_policy(name: str = BASE_MODEL, device: torch.device | str = "cpu", dtype=torch.float32):
    """The same architecture and size as `name`, with no pretraining.

    Build from the config. Calling model.apply(model._init_weights) after from_pretrained looks right but changes
    nothing in transformers 5: loaded tensors are flagged as initialised, and the init functions skip them.
    """
    return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), dtype=dtype).to(device)


def frozen_copy(model: torch.nn.Module) -> torch.nn.Module:
    """A reference model: same weights, eval mode, no gradients, never updated."""
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def tiny_llama(vocab_size: int = 128, seed: int = 0) -> LlamaForCausalLM:
    """A 2-layer random Llama (the SmolLM2 architecture, shrunk) for tests that only need the plumbing."""
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=True,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    cfg._attn_implementation = "sdpa"
    return LlamaForCausalLM(cfg).eval()
