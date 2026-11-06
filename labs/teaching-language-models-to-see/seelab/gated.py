"""Flamingo's bridge, small: tanh-gated cross-attention layers inserted into a frozen language model.

Each inserted block computes, for text hidden states x and visual features V:
    x = x + tanh(a_attn) * CrossAttn(LN(x), V)
    x = x + tanh(a_ff)   * FFW(LN(x))
and then the frozen language-model layer runs as before. a_attn and a_ff start at 0, so at step 0 the model
is exactly the original language model. Flamingo also has a Perceiver Resampler in front; we leave it out and
cross-attend to the 49 pixel-shuffled SigLIP tokens directly.
"""

from __future__ import annotations

import torch
from torch import nn

from . import common  # noqa: F401  (puts C7 on sys.path)
from tinyvlm.projector import pixel_shuffle  # noqa: E402


class GatedCrossAttention(nn.Module):
    def __init__(self, d: int, d_vis: int, heads: int, ff_mult: int = 2):
        super().__init__()
        self.kv = nn.Linear(d_vis, d)
        self.ln_attn, self.ln_ff = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.alpha_attn = nn.Parameter(torch.zeros(()))
        self.alpha_ff = nn.Parameter(torch.zeros(()))
        self.last_ratio: float | None = None  # |what the block adds| / |residual stream|, from the last forward

    def forward(self, x: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        x0 = x
        kv = self.kv(visual)
        a, _ = self.attn(self.ln_attn(x), kv, kv, need_weights=False)
        x = x + torch.tanh(self.alpha_attn) * a
        x = x + torch.tanh(self.alpha_ff) * self.ff(self.ln_ff(x))
        with torch.no_grad():
            self.last_ratio = ((x - x0).norm() / x0.norm()).item()
        return x


class GatedLayer(nn.Module):
    """A frozen decoder layer with a gated cross-attention block in front of it."""

    def __init__(self, xattn: GatedCrossAttention, layer: nn.Module):
        super().__init__()
        self.xattn, self.layer = xattn, layer
        self.visual: torch.Tensor | None = None
        if hasattr(layer, "attention_type"):
            self.attention_type = layer.attention_type

    def forward(self, hidden_states, *args, **kwargs):
        if self.visual is not None:
            hidden_states = self.xattn(hidden_states, self.visual.to(hidden_states.dtype))
        return self.layer(hidden_states, *args, **kwargs)


class FlamingoLite(nn.Module):
    """Frozen vision encoder + frozen language model + trainable gated cross-attention every `every` layers."""

    def __init__(self, vision: nn.Module, lm: nn.Module, every: int = 6, shuffle: int = 2, heads: int = 9):
        super().__init__()
        self.vision, self.lm, self.shuffle = vision, lm, shuffle
        d, d_vis = lm.config.hidden_size, vision.config.hidden_size * shuffle ** 2
        layers = lm.model.layers
        self.gated: list[GatedLayer] = []
        for i in range(0, len(layers), every):
            g = GatedLayer(GatedCrossAttention(d, d_vis, heads), layers[i])
            layers[i] = g
            self.gated.append(g)
        self.vision.requires_grad_(False)
        self.lm.requires_grad_(False)
        for g in self.gated:
            g.xattn.requires_grad_(True)

    def trainable(self) -> list[nn.Parameter]:
        return [p for g in self.gated for p in g.xattn.parameters()]

    def gates(self) -> list[tuple[float, float]]:
        return [(torch.tanh(g.xattn.alpha_attn).item(), torch.tanh(g.xattn.alpha_ff).item()) for g in self.gated]

    def contributions(self) -> list[float]:
        return [g.xattn.last_ratio for g in self.gated]

    def visual_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.vision(pixel_values=pixel_values).last_hidden_state
        return pixel_shuffle(feats, self.shuffle)

    def _set(self, visual):
        for g in self.gated:
            g.visual = visual

    def forward(self, input_ids, attention_mask=None, labels=None, pixel_values=None):
        self._set(None if pixel_values is None else self.visual_features(pixel_values))
        try:
            return self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            self._set(None)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values, **kw):
        self._set(self.visual_features(pixel_values))
        try:
            return self.lm.generate(input_ids=input_ids, attention_mask=attention_mask, **kw)[:, input_ids.shape[1]:]
        finally:
            self._set(None)
