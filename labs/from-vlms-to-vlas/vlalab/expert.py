"""A pi0-style action expert: two sets of transformer weights that share self-attention.

Prefix tokens (image and text) go through the backbone weights. Suffix tokens (robot state and noisy actions) go
through the smaller expert weights. In every layer all tokens attend jointly, under a block-wise causal mask with
three blocks: [image + text] [state] [actions]. A block sees itself and the blocks before it, never later ones.

Because the prefix never attends to the suffix, its keys and values can be computed once per observation and reused
for every flow-matching step. `stop_grad=True` detaches those cached keys and values when the expert reads them:
the knowledge-insulation stop-gradient.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def block_causal_mask(sizes: list[int]) -> torch.Tensor:
    """Boolean (N, N) mask, True where attention is allowed: query block >= key block."""
    block = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])
    return block[:, None] >= block[None, :]


class Weights(nn.Module):
    """One expert's weights for one layer: norms, QKV and output projections, and an MLP."""

    def __init__(self, d: int, heads: int, head_dim: int, mlp: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * heads * head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))


class MoTLayer(nn.Module):
    def __init__(self, d_vlm: int, d_exp: int, heads: int, head_dim: int, mlp_vlm: int, mlp_exp: int):
        super().__init__()
        self.h, self.hd = heads, head_dim
        self.vlm = Weights(d_vlm, heads, head_dim, mlp_vlm)
        self.exp = Weights(d_exp, heads, head_dim, mlp_exp)

    def qkv(self, w: Weights, x):
        b, n, _ = x.shape
        q, k, v = w.qkv(w.n1(x)).view(b, n, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        return q, k, v  # each (b, heads, n, head_dim)

    def finish(self, w: Weights, x, attn):
        b, _, n, _ = attn.shape
        x = x + w.out(attn.transpose(1, 2).reshape(b, n, -1))
        return x + w.mlp(w.n2(x))

    def prefix(self, x, mask):
        q, k, v = self.qkv(self.vlm, x)
        return self.finish(self.vlm, x, F.scaled_dot_product_attention(q, k, v, attn_mask=mask)), (k, v)

    def suffix(self, y, kv, mask, stop_grad: bool):
        """Suffix tokens attend to the cached prefix keys and values and to each other."""
        q, k, v = self.qkv(self.exp, y)
        pk, pv = kv
        if stop_grad:
            pk, pv = pk.detach(), pv.detach()
        attn = F.scaled_dot_product_attention(q, torch.cat([pk, k], 2), torch.cat([pv, v], 2), attn_mask=mask)
        return self.finish(self.exp, y, attn)


class ActionExpertPolicy(nn.Module):
    def __init__(self, act_dim: int, state_dim: int, horizon: int, layers: int = 4, d_vlm: int = 256,
                 d_exp: int = 128, heads: int = 4, head_dim: int = 64, mlp_vlm: int = 1024, mlp_exp: int = 256):
        super().__init__()
        self.act_dim, self.horizon = act_dim, horizon
        self.layers = nn.ModuleList(MoTLayer(d_vlm, d_exp, heads, head_dim, mlp_vlm, mlp_exp) for _ in range(layers))
        self.state_in = nn.Linear(state_dim, d_exp)
        self.action_in = nn.Linear(act_dim, d_exp)
        self.time_in = nn.Sequential(nn.Linear(2 * 8, d_exp), nn.SiLU(), nn.Linear(d_exp, d_exp))
        self.pos = nn.Parameter(torch.randn(horizon, d_exp) * 0.02)
        self.action_out = nn.Linear(d_exp, act_dim)
        self.final_norm = nn.LayerNorm(d_exp)

    # ---- prefix: run once per observation
    def encode_prefix(self, prefix):
        """prefix: (b, P, d_vlm) embeddings. Returns per-layer (k, v) and the final prefix hidden states."""
        mask = block_causal_mask([prefix.shape[1]])
        cache, x = [], prefix
        for layer in self.layers:
            x, kv = layer.prefix(x, mask)
            cache.append(kv)
        return cache, x

    # ---- suffix: run once per flow step
    def velocity(self, cache, state, a_tau, tau, stop_grad: bool = False):
        b = a_tau.shape[0]
        freqs = 2.0 ** torch.arange(8, dtype=tau.dtype) * math.pi
        t = self.time_in(torch.cat([torch.sin(tau[:, None] * freqs), torch.cos(tau[:, None] * freqs)], -1))
        y = torch.cat([self.state_in(state)[:, None], self.action_in(a_tau) + self.pos + t[:, None]], 1)
        P = cache[0][0].shape[2]
        mask = block_causal_mask([P, 1, self.horizon])[P:]  # rows for the suffix queries only
        for layer, kv in zip(self.layers, cache):
            y = layer.suffix(y, kv, mask, stop_grad)
        return self.action_out(self.final_norm(y[:, 1:]))

    def flow_loss(self, prefix, state, actions, stop_grad: bool = False, cache=None):
        if cache is None:
            cache, _ = self.encode_prefix(prefix)
        eps = torch.randn_like(actions)
        u = torch.distributions.Beta(1.5, 1.0).sample((len(actions),))
        tau = 0.999 * (1 - u)
        a_tau = tau[:, None, None] * actions + (1 - tau[:, None, None]) * eps
        return F.mse_loss(self.velocity(cache, state, a_tau, tau, stop_grad), actions - eps)

    @torch.no_grad()
    def sample(self, prefix, state, steps: int = 10, generator=None, cache=None):
        if cache is None:
            cache, _ = self.encode_prefix(prefix)
        a = torch.randn(len(state), self.horizon, self.act_dim, generator=generator)
        for i in range(steps):
            tau = torch.full((len(state),), i / steps)
            a = a + self.velocity(cache, state, a, tau) / steps
        return a

    # ---- reference: everything in one pass, with the full (P + 1 + H) mask
    def full_forward(self, prefix, state, a_tau, tau):
        """Same computation without a cache: one joint sequence and one mask. Used to check the cache is exact."""
        P = prefix.shape[1]
        freqs = 2.0 ** torch.arange(8, dtype=tau.dtype) * math.pi
        t = self.time_in(torch.cat([torch.sin(tau[:, None] * freqs), torch.cos(tau[:, None] * freqs)], -1))
        y = torch.cat([self.state_in(state)[:, None], self.action_in(a_tau) + self.pos + t[:, None]], 1)
        x = prefix
        mask = block_causal_mask([P, 1, self.horizon])
        for layer in self.layers:
            qx, kx, vx = layer.qkv(layer.vlm, x)
            qy, ky, vy = layer.qkv(layer.exp, y)
            q, k, v = (torch.cat(z, 2) for z in ((qx, qy), (kx, ky), (vx, vy)))
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = layer.finish(layer.vlm, x, attn[:, :, :P])
            y = layer.finish(layer.exp, y, attn[:, :, P:])
        return self.action_out(self.final_norm(y[:, 1:])), x

    def param_split(self) -> dict:
        vlm = sum(p.numel() for l in self.layers for p in l.vlm.parameters())
        return {"backbone": vlm, "expert": sum(p.numel() for p in self.parameters()) - vlm}
