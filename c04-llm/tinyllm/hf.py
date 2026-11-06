"""Hugging Face compatibility: TinyLLM <-> transformers.LlamaForCausalLM.

The architectures are the same, and TinyLLM uses Llama's parameter names, so conversion is
a config translation plus a "model." prefix on every key except lm_head.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import ModelConfig
from .model import TinyLLM


def to_hf_config(c: ModelConfig):
    from transformers import LlamaConfig

    if c.num_experts:
        raise ValueError("MoE blocks have no LlamaForCausalLM equivalent")
    return LlamaConfig(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        num_hidden_layers=c.num_layers,
        num_attention_heads=c.num_heads,
        num_key_value_heads=c.num_kv_heads,
        head_dim=c.head_dim,
        max_position_embeddings=c.max_seq_len,
        rms_norm_eps=c.rms_norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": c.rope_theta},
        tie_word_embeddings=c.tie_word_embeddings,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
    )


def from_hf_config(hf) -> ModelConfig:
    rope = getattr(hf, "rope_parameters", None) or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"rope_type {rope['rope_type']!r} (scaled RoPE) is not implemented in tinyllm")
    if getattr(hf, "attention_bias", False) or getattr(hf, "mlp_bias", False):
        raise ValueError("biases are not implemented in tinyllm")
    return ModelConfig(
        vocab_size=hf.vocab_size,
        hidden_size=hf.hidden_size,
        num_layers=hf.num_hidden_layers,
        num_heads=hf.num_attention_heads,
        num_kv_heads=hf.num_key_value_heads,
        intermediate_size=hf.intermediate_size,
        max_seq_len=hf.max_position_embeddings,
        head_dim=getattr(hf, "head_dim", None),
        rope_theta=float(rope.get("rope_theta", getattr(hf, "rope_theta", 10_000.0))),
        rms_norm_eps=hf.rms_norm_eps,
        tie_word_embeddings=hf.tie_word_embeddings,
    )


def to_hf_state_dict(model: TinyLLM) -> dict[str, torch.Tensor]:
    """Rename keys: 'layers.0.self_attn.q_proj.weight' -> 'model.layers.0.self_attn.q_proj.weight'."""
    out = {}
    for name, tensor in model.state_dict().items():
        out[name if name.startswith("lm_head.") else f"model.{name}"] = tensor
    return out


def to_hf(model: TinyLLM):
    """Build a LlamaForCausalLM carrying exactly the same weights."""
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM(to_hf_config(model.config))
    missing, unexpected = hf.load_state_dict(to_hf_state_dict(model), strict=False)
    allowed = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    if set(missing) - allowed or unexpected:
        raise RuntimeError(f"HF conversion mismatch: missing={missing}, unexpected={unexpected}")
    return hf.to(next(model.parameters()).dtype).eval()


def from_hf(hf_model) -> TinyLLM:
    """Copy a LlamaForCausalLM's weights into a fresh TinyLLM."""
    config = from_hf_config(hf_model.config)
    model = TinyLLM(config)
    state = {}
    for name, tensor in hf_model.state_dict().items():
        state[name.removeprefix("model.")] = tensor
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"HF import mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(next(hf_model.parameters()).dtype).eval()


def save_hf(model: TinyLLM, path: str | Path) -> None:
    """Write a folder that AutoModelForCausalLM.from_pretrained can open."""
    to_hf(model).save_pretrained(str(path))


def from_pretrained(name_or_path: str, dtype: torch.dtype = torch.float32) -> TinyLLM:
    """Load any default-RoPE Llama checkpoint from the Hub (for example SmolLM2-135M) into TinyLLM."""
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype)
    return from_hf(hf)
