"""KV caches: keep past keys and values so decoding does not recompute them.

Both caches store, per layer, keys (B, Hkv, T, d) and values (B, Hkv, T, dv),
and expose the same two methods the attention layers call:
    get_seq_length(layer_idx) -> T, the number of cached positions
    update(k, v, layer_idx)   -> all keys and values so far, new ones appended
"""

from __future__ import annotations

import torch


class KVCache:
    """Dynamic cache. Grows with torch.cat, so every step reallocates. Simple and fine for learning."""

    def __init__(self) -> None:
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.keys[layer_idx].size(-2) if layer_idx < len(self.keys) else 0

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == len(self.keys):  # first call for this layer: prefill
            self.keys.append(k)
            self.values.append(v)
        else:  # decode: append along the sequence dim
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], k], dim=-2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], v], dim=-2)
        return self.keys[layer_idx], self.values[layer_idx]

    def nbytes(self) -> int:
        """Bytes actually held by the cache tensors."""
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)


class StaticKVCache:
    """Preallocated cache. One buffer of max_seq_len, written in place.

    Memory is reserved up front, no reallocation happens while decoding, and the
    tensor addresses never change (which is what CUDA graphs need).
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        v_head_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        v_head_dim = v_head_dim or head_dim
        self.k = torch.zeros(num_layers, batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros(num_layers, batch_size, num_kv_heads, max_seq_len, v_head_dim, dtype=dtype, device=device)
        self.lengths = [0] * num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.lengths[layer_idx]

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        start, n = self.lengths[layer_idx], k.size(-2)
        if start + n > self.k.size(3):
            raise RuntimeError(f"cache full: {start + n} > max_seq_len {self.k.size(3)}")
        self.k[layer_idx, :, :, start : start + n] = k
        self.v[layer_idx, :, :, start : start + n] = v
        self.lengths[layer_idx] = start + n
        # Views of the filled prefix. Production code returns the whole buffer plus a mask
        # so shapes stay fixed; slicing keeps this version easy to check.
        return self.k[layer_idx, :, :, : start + n], self.v[layer_idx, :, :, : start + n]

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()


def kv_cache_bytes(
    num_layers: int, batch_size: int, seq_len: int, num_kv_heads: int, head_dim: int, bytes_per_elem: int
) -> int:
    """M_KV = 2 * L * B * S * H_kv * d_h * b. The 2 is one tensor for K and one for V."""
    return 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * bytes_per_elem


def mla_cache_bytes(
    num_layers: int, batch_size: int, seq_len: int, kv_lora_rank: int, rope_head_dim: int, bytes_per_elem: int
) -> int:
    """MLA caches one latent c_kv (r) plus one shared RoPE key (d_r) per token: L * B * S * (r + d_r) * b."""
    return num_layers * batch_size * seq_len * (kv_lora_rank + rope_head_dim) * bytes_per_elem
