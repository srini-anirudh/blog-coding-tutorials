"""Read and edit a VLM's internals: attention weights, attention knockout, residual capture and patching, and
embedding ablation.

Everything here works through one `Intervention` object. It is consulted by a custom attention function and by
forward hooks, so a single forward pass can record attention, block routes and patch activations at once.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface, eager_mask

NAME = "vlmscope_eager"


@dataclass
class Intervention:
    record_attention: bool = False
    # layer -> list of (target positions, source positions). Scores from target to source become -inf.
    knockout: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    # layer -> (positions, replacement [len(positions), d]). Overwrites the residual stream after that layer.
    patch: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    # (positions, replacement) written into the input embeddings before layer 0.
    embed_patch: tuple[torch.Tensor, torch.Tensor] | None = None
    capture_residual: bool = False
    # filled during the forward pass
    attentions: dict[int, torch.Tensor] = field(default_factory=dict)   # layer -> [B, heads, q, k]
    residual: dict[int, torch.Tensor] = field(default_factory=dict)     # layer -> [B, seq, d]; -1 = embeddings


_ACTIVE: list[Intervention] = []


def _active() -> Intervention | None:
    return _ACTIVE[-1] if _ACTIVE else None


def lab_attention(module: nn.Module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention with two extra powers: record the weights and knock out chosen routes.

    a_ij = softmax_j(q_i . k_j / sqrt(d)). Knockout sets the score to -inf before the softmax, so position i
    cannot read position j at this layer (Geva et al., 2023).
    """
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]
    iv = _active()
    layer = getattr(module, "_vlmscope_layer", None)
    if iv is not None and layer is not None and layer in iv.knockout:
        block = torch.zeros(scores.shape[-2:], dtype=torch.bool, device=scores.device)
        for tgt, src in iv.knockout[layer]:
            block[tgt.to(block.device).unsqueeze(1), src.to(block.device).unsqueeze(0)] = True
        scores = scores.masked_fill(block, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if iv is not None and layer is not None and iv.record_attention:
        iv.attentions[layer] = weights.detach()
    out = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return out, weights


AttentionInterface.register(NAME, lab_attention)
if NAME not in ALL_MASK_ATTENTION_FUNCTIONS:
    AttentionMaskInterface.register(NAME, eager_mask)


def install(text_model: nn.Module) -> list:
    """Tag each decoder layer's attention with its index and add residual hooks. Returns hook handles."""
    handles = []
    for i, layer in enumerate(text_model.layers):
        layer.self_attn._vlmscope_layer = i
        handles.append(layer.register_forward_hook(_make_residual_hook(i)))
    handles.append(text_model.register_forward_pre_hook(_embed_hook, with_kwargs=True))
    return handles


def _make_residual_hook(i: int):
    def hook(module, args, output):
        iv = _active()
        if iv is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        if i in iv.patch:
            pos, rep = iv.patch[i]
            h = h.clone()
            h[:, pos.to(h.device)] = rep.to(h.device, h.dtype)
        if iv.capture_residual:
            iv.residual[i] = h.detach()
        if i in iv.patch:
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return None
    return hook


def _embed_hook(module, args, kwargs):
    iv = _active()
    if iv is None:
        return None
    emb = kwargs.get("inputs_embeds")
    if emb is None:
        return None
    if iv.embed_patch is not None:
        pos, rep = iv.embed_patch
        emb = emb.clone()
        emb[:, pos.to(emb.device)] = rep.to(emb.device, emb.dtype)
        kwargs["inputs_embeds"] = emb
    if iv.capture_residual:
        iv.residual[-1] = emb.detach()
    return args, kwargs


@contextmanager
def intervene(iv: Intervention):
    _ACTIVE.append(iv)
    try:
        yield iv
    finally:
        _ACTIVE.pop()
