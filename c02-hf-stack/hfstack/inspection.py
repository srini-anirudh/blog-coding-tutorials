"""Print what is inside configs, model outputs and caches, so nothing is a black box."""

from __future__ import annotations

from typing import Any

import torch
from transformers import Cache, PreTrainedConfig

ARCHITECTURE_FIELDS = [
    "model_type",
    "architectures",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "rope_parameters",
    "tie_word_embeddings",
    "dtype",
]


def config_fields(config: PreTrainedConfig, fields: list[str] = ARCHITECTURE_FIELDS) -> dict[str, Any]:
    """The fields that decide every tensor shape in the model."""
    return {name: getattr(config, name, None) for name in fields}


def shape_of(value: Any) -> str:
    """A short description of a tensor, a tuple of tensors, or a cache."""
    if isinstance(value, torch.Tensor):
        return f"{tuple(value.shape)} {str(value.dtype).replace('torch.', '')}"
    if isinstance(value, Cache):
        layer = value.layers[0]
        return (
            f"{type(value).__name__}: {len(value.layers)} layers, "
            f"keys {tuple(layer.keys.shape)}, values {tuple(layer.values.shape)}"
        )
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return f"tuple of {len(value)} × {tuple(value[0].shape)}"
    return type(value).__name__


def describe_outputs(outputs) -> dict[str, str]:
    """Every non-empty field of a ModelOutput and its shape."""
    return {key: shape_of(value) for key, value in outputs.items() if value is not None}
