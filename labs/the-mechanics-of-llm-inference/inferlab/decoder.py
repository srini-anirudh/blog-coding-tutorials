"""A Llama-shaped decoder built for measuring inference, with a preallocated KV cache.

C4's tinyllm (c04-llm/tinyllm) is the teaching model: it grows its cache with torch.cat, which changes tensor
shapes every step. This decoder keeps the same maths and the same Hugging Face weight names, but writes keys and
values into one fixed buffer. Fixed shapes give three things the measurements need:

- a decode step's bytes are exactly the weights plus the cache slice it reads, with no copies;
- a decode step can be captured once as a CUDA graph and replayed, which removes Python and launch overhead;
- rolling back rejected speculative tokens is just moving the write position back.

tests/test_decoder.py checks that it gives the same logits as tinyllm and as Hugging Face LlamaForCausalLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DecoderConfig:
    vocab: int
    d_model: int
    layers: int
    n_q: int
    n_kv: int
    ffn: int
    rope_theta: float = 10_000.0
    tied: bool = False
    eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_q


SHAPES = {
    # Public configs of the real models. Only the shape is used; weights are random.
    "llama-3-8b": DecoderConfig(128_256, 4096, 32, 32, 8, 14_336, 500_000.0),
    "llama-2-7b": DecoderConfig(32_000, 4096, 32, 32, 32, 11_008, 10_000.0),
    "smollm2-135m": DecoderConfig(49_152, 576, 30, 9, 3, 1536, 100_000.0, tied=True),
    "tiny": DecoderConfig(256, 128, 4, 4, 2, 352),
}


class Attention(nn.Module):
    def __init__(self, c: DecoderConfig) -> None:
        super().__init__()
        hd = c.head_dim
        self.q_proj = nn.Linear(c.d_model, c.n_q * hd, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_kv * hd, bias=False)
        self.o_proj = nn.Linear(c.n_q * hd, c.d_model, bias=False)


class MLP(nn.Module):
    def __init__(self, c: DecoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(c.d_model, c.ffn, bias=False)
        self.up_proj = nn.Linear(c.d_model, c.ffn, bias=False)
        self.down_proj = nn.Linear(c.ffn, c.d_model, bias=False)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()  # the mean of squares in float32, as Llama does
        return self.weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class Block(nn.Module):
    def __init__(self, c: DecoderConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(c.d_model, c.eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Decoder(nn.Module):
    """input ids (B, T) written at positions [start, start + T) -> logits.

    The cache holds K and V for every layer: (L, B, n_kv, max_len, d_head) each. `start` is a Python int, so a
    captured CUDA graph is tied to one position. That is what a benchmark wants: one graph per context length.
    """

    def __init__(self, c: DecoderConfig, max_len: int = 2048) -> None:
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab, c.d_model)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab, bias=False)
        if c.tied:
            self.lm_head.weight = self.embed_tokens.weight
        self.max_len = max_len
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    # ---- construction --------------------------------------------------------------------------------------

    @classmethod
    def random(cls, c: DecoderConfig, max_len: int, device: str = "cpu", dtype=torch.float32,
               seed: int = 0) -> "Decoder":
        """Random weights made straight on the device (an 8B model in bf16 builds in about a second on a GPU)."""
        with torch.device("meta"):
            model = cls(c, max_len)
        model.to_empty(device=device)
        model._tie()  # to_empty makes a fresh tensor per module, which would untie the LM head
        gen = torch.Generator(device=device).manual_seed(seed)
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                p.data.fill_(1.0)
            else:
                p.data.normal_(0.0, 0.02, generator=gen)
        model.to(dtype)
        model._tie()
        model._build_rope(device)
        return model.eval()

    @classmethod
    def from_hf(cls, model_dir: str | Path, max_len: int = 2048, device: str = "cpu",
                dtype=torch.float32) -> "Decoder":
        """Load a Llama-architecture checkpoint (config.json + model.safetensors) from a local folder."""
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        c = DecoderConfig(cfg["vocab_size"], cfg["hidden_size"], cfg["num_hidden_layers"],
                          cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["intermediate_size"],
                          float(cfg.get("rope_theta", 10_000.0)), bool(cfg.get("tie_word_embeddings", False)),
                          float(cfg.get("rms_norm_eps", 1e-5)))
        state = {k.removeprefix("model."): v for k, v in load_file(str(model_dir / "model.safetensors")).items()}
        with torch.device("meta"):
            model = cls(c, max_len)
        model.load_state_dict(state, strict=False, assign=True)
        model._tie()
        missing = [n for n, p in model.named_parameters() if p.is_meta]
        if missing:
            raise ValueError(f"checkpoint is missing {missing[:3]}")
        model.to(device=device, dtype=dtype)
        model._tie()
        model._build_rope(device)
        return model.eval()

    def _tie(self) -> None:
        if self.config.tied:
            self.lm_head.weight = self.embed_tokens.weight

    def _build_rope(self, device) -> None:
        hd = self.config.head_dim
        inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32, device=device) / hd))
        freqs = torch.outer(torch.arange(self.max_len, dtype=torch.float32, device=device), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.rope_cos, self.rope_sin = emb.cos(), emb.sin()  # (max_len, d_head), float32

    def allocate_cache(self, batch: int, max_len: int | None = None) -> int:
        """Preallocate K and V for `batch` sequences. Returns the bytes allocated."""
        c, p = self.config, self.lm_head.weight
        if max_len is not None and max_len != self.max_len:
            self.max_len = max_len
            self._build_rope(p.device)
        self.k_cache = self.v_cache = None  # free the old buffers first
        shape = (c.layers, batch, c.n_kv, self.max_len, c.head_dim)
        self.k_cache = torch.zeros(shape, dtype=p.dtype, device=p.device)
        self.v_cache = torch.zeros(shape, dtype=p.dtype, device=p.device)
        return 2 * self.k_cache.numel() * self.k_cache.element_size()

    def weight_bytes(self, read_per_decode_step: bool = False) -> int:
        """All weight bytes, or only those a decode step reads (the embedding table is gathered, not streamed)."""
        total = sum(p.numel() * p.element_size() for p in self.parameters())
        if read_per_decode_step and not self.config.tied:
            total -= self.embed_tokens.weight.numel() * self.embed_tokens.weight.element_size()
        return total

    # ---- forward ---------------------------------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, start: int, last_only: bool = True) -> torch.Tensor:
        """ids (B, T) -> logits (B, 1, V) if last_only else (B, T, V). Writes K, V at [start, start + T)."""
        if self.k_cache is None:
            raise RuntimeError("call allocate_cache(batch) first")
        c = self.config
        b, t = ids.shape
        end = start + t
        if end > self.max_len:
            raise ValueError(f"position {end} exceeds the cache length {self.max_len}")
        x = self.embed_tokens(ids)
        cos = self.rope_cos[start:end].to(x.dtype)  # (T, d_head)
        sin = self.rope_sin[start:end].to(x.dtype)
        mask = None
        if start > 0 and t > 1:
            # New row i sits at position start + i and may see keys 0 .. start + i.
            mask = torch.ones(t, end, dtype=torch.bool, device=ids.device).tril(diagonal=start)
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(b, t, c.n_q, c.head_dim).transpose(1, 2)  # (B, n_q, T, d_head)
            k = a.k_proj(h).view(b, t, c.n_kv, c.head_dim).transpose(1, 2)  # (B, n_kv, T, d_head)
            v = a.v_proj(h).view(b, t, c.n_kv, c.head_dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin  # keys are cached already rotated
            self.k_cache[i, :b, :, start:end] = k
            self.v_cache[i, :b, :, start:end] = v
            keys, values = self.k_cache[i, :b, :, :end], self.v_cache[i, :b, :, :end]  # the whole past
            if start == 0:
                o = F.scaled_dot_product_attention(q, keys, values, is_causal=True, enable_gqa=True)
            else:
                o = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, enable_gqa=True)
            x = x + a.o_proj(o.transpose(1, 2).reshape(b, t, c.n_q * c.head_dim))
            h = layer.post_attention_layernorm(x)
            x = x + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(h)) * layer.mlp.up_proj(h))
        if last_only:
            x = x[:, -1:]  # the prefill logits trap: only the last position needs the LM head
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def forward_no_cache(self, ids: torch.Tensor) -> torch.Tensor:
        """Run the whole sequence from position 0, as generation without a cache must. Logits (B, T, V)."""
        return self.forward(ids, 0, last_only=False)


@torch.no_grad()
def greedy_generate(model: Decoder, prompt: torch.Tensor, n_new: int, use_cache: bool = True) -> torch.Tensor:
    """Greedy decoding with the cache (prefill once, then one token per step) or without it (re-run the prefix)."""
    ids = prompt
    for step in range(n_new):
        if not use_cache:
            logits = model.forward(ids, 0)
        elif step == 0:
            logits = model.forward(ids, 0)  # prefill
        else:
            logits = model.forward(ids[:, -1:], ids.shape[1] - 1)  # decode: one new position
        ids = torch.cat((ids, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    return ids
