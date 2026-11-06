"""The single-device reference: a pre-norm transformer block on [seq, batch, hidden] activations.

Sequence-first layout, as in Megatron, so that sequence parallelism splits dimension 0.
Every parallel version in this package is checked against this block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, causal: bool = True):
    """q, k, v: [batch, heads, seq, head_dim]. Plain softmax attention, materialising the scores."""
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        n_q, n_k = s.shape[-2:]
        s = s.masked_fill(torch.ones(n_q, n_k, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    return s.softmax(-1) @ v


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, ffn_mult: int = 4):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, ffn_mult * hidden)
        self.fc2 = nn.Linear(ffn_mult * hidden, hidden)

    def attn(self, x):
        s, b, _ = x.shape
        q, k, v = self.qkv(x).view(s, b, self.heads, 3, -1).permute(3, 1, 2, 0, 4)  # 3 x [b, heads, s, d]
        o = attention(q, k, v).permute(2, 0, 1, 3).reshape(s, b, -1)
        return self.proj(o)

    def mlp(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Stack(nn.Module):
    """A small model for the pipeline: input projection, L blocks, output head."""

    def __init__(self, hidden: int, heads: int, layers: int, d_in: int = 16, d_out: int = 16):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList(Block(hidden, heads) for _ in range(layers))
        self.out = nn.Linear(hidden, d_out)

    def forward(self, x):
        x = self.inp(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)
