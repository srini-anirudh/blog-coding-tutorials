"""Key/value cache: keep every layer's past keys and values so decoding never recomputes them.

Same interface as C3's attnzoo.KVCache (and Hugging Face's DynamicCache):
    get_seq_length(layer_idx) -> number of cached positions
    update(k, v, layer_idx)   -> all keys and values so far, the new ones appended
"""

from __future__ import annotations

import torch


class KVCache:
    """Dynamic cache. Each entry is (B, H_kv, T, d) and grows by torch.cat along T.

    Keys are stored after RoPE, so a cached key never needs to be rotated again.
    Storing H_kv heads (not H) is where GQA saves memory.
    """

    def __init__(self) -> None:
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.keys[layer_idx].shape[2] if layer_idx < len(self.keys) else 0

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Append k, v (B, H_kv, T_new, d); return (B, H_kv, T_past + T_new, d)."""
        if layer_idx == len(self.keys):  # first call for this layer: prefill
            self.keys.append(k)
            self.values.append(v)
        else:  # decode: append along time
            self.keys[layer_idx] = torch.cat((self.keys[layer_idx], k), dim=2)
            self.values[layer_idx] = torch.cat((self.values[layer_idx], v), dim=2)
        return self.keys[layer_idx], self.values[layer_idx]

    def nbytes(self) -> int:
        """Bytes actually held by the cache tensors."""
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)
