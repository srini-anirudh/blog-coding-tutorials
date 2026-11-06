"""Where visual tokens meet the language model: four fusion points on one tiny decoder.

    prefix   visual tokens are placed in front of the text, the decoder is unchanged (LLaVA, Qwen-VL).
    xattn    the text stream stays text-only; tanh-gated cross-attention layers, zero-initialised,
             read the visual tokens (Flamingo, IDEFICS 1).
    expert   one stream, shared attention, but visual positions use their own QKV, output and FFN
             weights in every layer (CogVLM). With `expert_rank > 0` the visual weights are the text
             weights plus a low-rank delta used only at visual positions (InternLM-XComposer2's PLoRA).
    encfree  no vision encoder: raw patches are linearly projected and used as a prefix (Fuyu). This
             is a prefix decoder with a different front end, so it lives in `model.py`.

The decoder is a small pre-norm GPT with learned absolute positions, so every difference between
the modes is visible in the code below.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSIONS = ("prefix", "xattn", "expert", "encfree")


class RoutedLinear(nn.Module):
    """A linear layer whose weights can depend on whether a position is visual.

    mode None    : one weight for every position (a plain nn.Linear).
    mode "full"  : visual positions use a separate copy, initialised from the text weight (CogVLM).
    mode "lora"  : visual positions use W + B A, with B = 0 at init (PLoRA).
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.base = nn.Linear(d_in, d_out)
        self.mode = None
        self.expert = None
        self.lora_a = self.lora_b = None

    def add_expert(self, mode: str, rank: int = 0):
        self.mode = mode
        if mode == "full":
            self.expert = copy.deepcopy(self.base)
        elif mode == "lora":
            self.lora_a = nn.Linear(self.base.in_features, rank, bias=False)
            self.lora_b = nn.Linear(rank, self.base.out_features, bias=False)
            nn.init.normal_(self.lora_a.weight, std=1 / math.sqrt(self.base.in_features))
            nn.init.zeros_(self.lora_b.weight)
        else:
            raise ValueError(mode)

    def forward(self, x, vis_mask=None):
        y = self.base(x)
        if self.mode is None or vis_mask is None:
            return y
        if self.mode == "full":
            y_vis = self.expert(x)
        else:
            y_vis = y + self.lora_b(self.lora_a(x))
        return torch.where(vis_mask.unsqueeze(-1), y_vis, y)


class GatedCrossAttention(nn.Module):
    """Flamingo's GATED XATTN-DENSE block. tanh(0) = 0, so at init the block is the identity and
    the model is an exact copy of the frozen language model."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ff = nn.Parameter(torch.zeros(1))

    def forward(self, h, mem):
        h = h + torch.tanh(self.gate_attn) * self.attn(self.ln_attn(h), mem, mem, need_weights=False)[0]
        return h + torch.tanh(self.gate_ff) * self.ff(self.ln_ff(h))


class Block(nn.Module):
    """Pre-norm decoder block. Every projection is a RoutedLinear so the visual-expert mode can
    swap in visual weights without a second class."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = RoutedLinear(d, 3 * d)
        self.out = RoutedLinear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = RoutedLinear(d, 4 * d)
        self.fc2 = RoutedLinear(4 * d, d)
        self.xattn: GatedCrossAttention | None = None

    def projections(self):
        return [self.qkv, self.out, self.fc1, self.fc2]

    def forward(self, h, vis_mask=None, mem=None):
        if self.xattn is not None and mem is not None:
            h = self.xattn(h, mem)
        b, n, d = h.shape
        q, k, v = self.qkv(self.ln1(h), vis_mask).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # attention itself is shared
        h = h + self.out(a.transpose(1, 2).reshape(b, n, d), vis_mask)
        return h + self.fc2(F.gelu(self.fc1(self.ln2(h), vis_mask)), vis_mask)


class TinyLM(nn.Module):
    """A small decoder-only language model. `embed` maps token ids; `forward_embeds` takes vectors,
    so a VLM can put visual tokens into the stream."""

    def __init__(self, vocab: int, d: int = 64, layers: int = 4, heads: int = 4, max_len: int = 48):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward_embeds(self, h, vis_mask=None, mem=None):
        h = h + self.pos(torch.arange(h.shape[1], device=h.device))
        for blk in self.blocks:
            h = blk(h, vis_mask, mem)
        return self.head(self.ln_f(h))

    def forward(self, ids):
        return self.forward_embeds(self.embed(ids))

    # --- the structural edits each fusion mode makes to a pretrained LM -------------------------
    def add_cross_attention(self, every: int = 1, heads: int = 4) -> list[nn.Module]:
        """Insert gated cross-attention before every `every`-th block. Returns the new modules."""
        new = []
        for i, blk in enumerate(self.blocks):
            if i % every == 0:
                blk.xattn = GatedCrossAttention(self.d, heads)
                new.append(blk.xattn)
        return new

    def add_visual_experts(self, rank: int = 0) -> list[nn.Parameter]:
        """Give visual positions their own weights in every block. Returns the new parameters."""
        new = []
        for blk in self.blocks:
            for proj in blk.projections():
                proj.add_expert("lora" if rank else "full", rank)
                mods = [proj.expert] if proj.expert is not None else [proj.lora_a, proj.lora_b]
                for m in mods:
                    new.extend(m.parameters())
        return new


def attention_cost(n_text: int, n_vis: int, d: int, layers: int, fusion: str) -> dict:
    """Rough multiply-accumulate count of the decoder attention scores (QK^T and AV) per sequence,
    and the KV positions cached, for each fusion point. It ignores projections and FFNs."""
    if fusion in ("prefix", "expert", "encfree"):
        n = n_text + n_vis
        return {"seq_len": n, "attn_macs": 2 * layers * n * n * d, "kv_positions": layers * n}
    if fusion == "xattn":
        self_macs = 2 * layers * n_text * n_text * d
        cross_macs = 2 * layers * n_text * n_vis * d
        # visual keys and values are computed once per layer and cached like a prefix would be
        return {"seq_len": n_text, "attn_macs": self_macs + cross_macs, "kv_positions": layers * (n_text + n_vis)}
    raise ValueError(fusion)
