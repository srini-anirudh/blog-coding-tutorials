"""A Llama forward pass over a ragged batch, with a paged KV cache.

The weights live in C4's TinyLLM (c04-llm/tinyllm), which can load SmolLM2 from the Hub. This file only adds
a different forward pass, in the layout the post calls selective batching:

    tokens of every sequence, flattened      x: (N, D)      no padding anywhere
    linear layers, MLP, norms                one GEMM over all N rows
    attention                                per sequence, reading its KV through a block table

Decode rows (one query each) are attended together in one batched gather. Prefill rows (a whole prompt or
one chunk of it) are attended one sequence at a time. The gather is plain PyTorch indexing, not a fused
paged-attention kernel, so it pays the extra memory traffic the post says hand-written kernels avoid.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

C4_DIR = Path(__file__).resolve().parents[3] / "c04-llm"
if str(C4_DIR) not in sys.path:
    sys.path.insert(0, str(C4_DIR))

from tinyllm import ModelConfig, TinyLLM, get_config  # noqa: E402
from tinyllm.rope import rotate_half  # noqa: E402


def kv_bytes_per_token(config: ModelConfig, dtype_bytes: int) -> int:
    """2 (K and V) x layers x KV heads x head_dim x bytes. Llama-3-8B in bf16: 128 KiB."""
    return 2 * config.num_layers * config.num_kv_heads * config.head_dim * dtype_bytes


class PagedKVCache:
    """k, v: (layers, num_blocks + 1, block_size, H_kv, d). The extra last block is scratch space that
    padded rows write into (CUDA graphs pad the batch up to a captured size)."""

    def __init__(self, config: ModelConfig, num_blocks: int, block_size: int, dtype, device) -> None:
        shape = (config.num_layers, num_blocks + 1, block_size, config.num_kv_heads, config.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.scratch_slot = num_blocks * block_size  # first slot of the scratch block

    def copy_block(self, src: int, dst: int) -> None:
        self.k[:, dst].copy_(self.k[:, src])
        self.v[:, dst].copy_(self.v[:, src])

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


@dataclass
class BatchMeta:
    """Everything one forward pass needs to know about the ragged batch. Built by the engine."""

    input_ids: torch.Tensor  # (N,)
    positions: torch.Tensor  # (N,) absolute position of each token in its own sequence
    slot_mapping: torch.Tensor  # (N,) physical slot = block * block_size + offset, where its K and V go
    logits_rows: torch.Tensor  # (S,) rows whose logits we need (last row of each sequence that samples)
    dec_rows: torch.Tensor | None = None  # (Bd,) rows that are single-token decodes
    dec_tables: torch.Tensor | None = None  # (Bd, max_blocks) block tables, padded with 0
    dec_ctx: torch.Tensor | None = None  # (Bd,) context length after this token is written
    prefills: list = field(default_factory=list)  # (row_start, row_end, ctx_len, slots (ctx,), mask (t, ctx) or None)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (N, H, d), cos/sin: (N, d)."""
    return x * cos[:, None] + rotate_half(x) * sin[:, None]


def paged_attention(q, k_cache, v_cache, meta: BatchMeta, dec_mask) -> torch.Tensor:
    """q: (N, H, d). k_cache, v_cache: (num_blocks + 1, block_size, H_kv, d) for one layer. Returns (N, H, d).

    Index tensors and masks are built once per forward pass (in PagedLlama.forward / the engine), not once
    per layer, because on a small model the Python and launch cost of building them would dominate.
    """
    out = torch.empty_like(q)
    hkv, hd = k_cache.shape[2], k_cache.shape[3]
    n_rep = q.shape[1] // hkv
    if meta.dec_rows is not None and meta.dec_rows.numel():
        b, nb = meta.dec_tables.shape
        bs = k_cache.shape[1]
        # The pointer chase: gather each sequence's blocks wherever the free list put them.
        k = k_cache[meta.dec_tables].reshape(b, nb * bs, hkv, hd).transpose(1, 2)  # (B, H_kv, T, d)
        v = v_cache[meta.dec_tables].reshape(b, nb * bs, hkv, hd).transpose(1, 2)
        qd = q[meta.dec_rows][:, :, None, :]  # (B, H, 1, d)
        o = F.scaled_dot_product_attention(qd, k, v, attn_mask=dec_mask, enable_gqa=n_rep > 1)
        out[meta.dec_rows] = o[:, :, 0, :]
    kf, vf = k_cache.view(-1, hkv, hd), v_cache.view(-1, hkv, hd)
    for r0, r1, ctx, slots, mask in meta.prefills:
        k = kf[slots].transpose(0, 1)[None]  # (1, H_kv, ctx, d)
        v = vf[slots].transpose(0, 1)[None]
        qp = q[r0:r1].transpose(0, 1)[None]  # (1, H, t, d)
        # mask is None for a whole prompt (plain causal). A later chunk sees every earlier chunk's KV,
        # which is already in the cache, so it gets an explicit shifted mask.
        o = F.scaled_dot_product_attention(qp, k, v, attn_mask=mask, is_causal=mask is None, enable_gqa=n_rep > 1)
        out[r0:r1] = o[0].transpose(0, 1)
    return out


class PagedLlama(torch.nn.Module):
    """Wraps a TinyLLM and runs it over a flattened ragged batch."""

    def __init__(self, base: TinyLLM) -> None:
        super().__init__()
        self.base = base
        self.config = base.config

    @torch.no_grad()
    def forward(self, meta: BatchMeta, cache: PagedKVCache) -> torch.Tensor:
        m, c = self.base, self.config
        n = meta.input_ids.shape[0]
        x = m.embed_tokens(meta.input_ids)  # (N, D)
        cos = m.rotary.cos_cached[meta.positions].to(x.dtype)  # (N, d)
        sin = m.rotary.sin_cached[meta.positions].to(x.dtype)
        dec_mask = None
        if meta.dec_rows is not None and meta.dec_rows.numel():
            t = meta.dec_tables.shape[1] * cache.block_size
            dec_mask = (torch.arange(t, device=x.device)[None, :] < meta.dec_ctx[:, None])[:, None, None, :]
        for li, layer in enumerate(m.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = _rope(a.q_proj(h).view(n, c.num_heads, c.head_dim), cos, sin)
            k = _rope(a.k_proj(h).view(n, c.num_kv_heads, c.head_dim), cos, sin)
            v = a.v_proj(h).view(n, c.num_kv_heads, c.head_dim)
            kc, vc = cache.k[li], cache.v[li]
            kc.view(-1, c.num_kv_heads, c.head_dim)[meta.slot_mapping] = k  # write this step's KV
            vc.view(-1, c.num_kv_heads, c.head_dim)[meta.slot_mapping] = v
            o = paged_attention(q, kc, vc, meta, dec_mask)
            x = x + a.o_proj(o.reshape(n, c.num_heads * c.head_dim))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return m.lm_head(m.norm(x[meta.logits_rows]))  # (S, V)


def load_model(name: str, dtype=torch.float32, device="cpu", seed: int = 0) -> PagedLlama:
    """name: 'tiny' (random C4 preset, for tests) or a Hub id such as HuggingFaceTB/SmolLM2-360M."""
    if name == "llama3-8b-random":
        # Llama-3-8B shapes (32 layers, 8 KV heads, 128 KiB of KV per token) with random weights. Speed and
        # memory do not depend on the weight values, so this stands in for an 8B model where the post's
        # arithmetic is about 8B. Outputs are meaningless text.
        torch.manual_seed(seed)
        cfg = get_config("llama3-8b", max_seq_len=8192)
        with torch.device(device):
            prev = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                base = TinyLLM(cfg)
            finally:
                torch.set_default_dtype(prev)
        return PagedLlama(base.eval())
    if name == "tiny":
        torch.manual_seed(seed)
        base = TinyLLM(get_config("tiny", max_seq_len=8192))
    else:
        from tinyllm.hf import from_pretrained

        base = from_pretrained(name, dtype=dtype)
    return PagedLlama(base.to(device=device, dtype=dtype).eval())
