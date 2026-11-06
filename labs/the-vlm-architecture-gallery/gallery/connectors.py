"""The connector families from the gallery, as swappable modules.

Every connector has the same contract:

    forward(x, grid) -> tokens
    x      B x N x d_vis    patch features from the vision encoder, N = h * w, row-major
    grid   (h, w)           the patch grid
    tokens B x M x d_model  what the language model receives

`out_tokens(h, w)` says how many tokens M come out, without running anything. That number is the
"what does a picture cost" number of the post.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in: int, d_out: int) -> nn.Sequential:
    """The LLaVA-1.5 projector: Linear -> GELU -> Linear."""
    return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))


def pixel_shuffle(x: torch.Tensor, grid: tuple[int, int], s: int) -> torch.Tensor:
    """Fold each s x s block of neighbouring patches into the channel axis.

    B x (h*w) x d  ->  B x (h/s * w/s) x (s*s*d). Nothing is averaged, so nothing is lost: the
    four patches of a 2 x 2 block sit side by side in one longer vector.
    """
    b, n, d = x.shape
    h, w = grid
    if h * w != n or h % s or w % s:
        raise ValueError(f"grid {grid} does not fit {n} patches with factor {s}")
    x = x.view(b, h, w // s, s * d)            # merge s neighbouring columns
    x = x.permute(0, 2, 1, 3)                  # b x w/s x h x s*d
    x = x.reshape(b, w // s, h // s, s * s * d)  # merge s neighbouring rows
    x = x.permute(0, 2, 1, 3)                  # back to row-major on the coarse grid
    return x.reshape(b, (h // s) * (w // s), s * s * d)


class LinearConnector(nn.Module):
    """LLaVA, PaliGemma: one matrix per patch, 1 : 1."""

    def __init__(self, d_vis: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_vis, d_model)

    def out_tokens(self, h: int, w: int) -> int:
        return h * w

    def forward(self, x, grid):
        return self.proj(x)


class MLPConnector(nn.Module):
    """LLaVA-1.5, VILA: a two-layer MLP per patch, still 1 : 1."""

    def __init__(self, d_vis: int, d_model: int):
        super().__init__()
        self.proj = mlp(d_vis, d_model)

    def out_tokens(self, h: int, w: int) -> int:
        return h * w

    def forward(self, x, grid):
        return self.proj(x)


class PixelShuffleConnector(nn.Module):
    """InternVL 1.5+, Idefics3, DeepSeek-VL2, Qwen's 2 x 2 merge: fold, then MLP. Exactly / s^2."""

    def __init__(self, d_vis: int, d_model: int, s: int = 2):
        super().__init__()
        self.s = s
        self.norm = nn.LayerNorm(d_vis * s * s)
        self.proj = mlp(d_vis * s * s, d_model)

    def out_tokens(self, h: int, w: int) -> int:
        return (h // self.s) * (w // self.s)

    def forward(self, x, grid):
        return self.proj(self.norm(pixel_shuffle(x, grid, self.s)))


class AvgPoolConnector(nn.Module):
    """Gemma 3 style: average k x k neighbourhoods, then project. Exactly / k^2, but averaged."""

    def __init__(self, d_vis: int, d_model: int, k: int = 2):
        super().__init__()
        self.k = k
        self.proj = mlp(d_vis, d_model)

    def out_tokens(self, h: int, w: int) -> int:
        return (h // self.k) * (w // self.k)

    def forward(self, x, grid):
        b, n, d = x.shape
        h, w = grid
        img = x.transpose(1, 2).reshape(b, d, h, w)
        pooled = F.avg_pool2d(img, self.k).flatten(2).transpose(1, 2)  # b x (h/k * w/k) x d
        return self.proj(pooled)


class AttnPoolConnector(nn.Module):
    """Molmo style attentional pooling: in each k x k window, the window mean is the query and
    attends over the k^2 members. Learned weights decide what survives, instead of a plain mean."""

    def __init__(self, d_vis: int, d_model: int, k: int = 2, heads: int = 4):
        super().__init__()
        self.k = k
        self.attn = nn.MultiheadAttention(d_vis, heads, batch_first=True)
        self.proj = mlp(d_vis, d_model)

    def out_tokens(self, h: int, w: int) -> int:
        return (h // self.k) * (w // self.k)

    def forward(self, x, grid):
        b, n, d = x.shape
        k = self.k
        windows = pixel_shuffle(x, grid, k)                 # b x M x (k*k*d), one row per window
        m = windows.shape[1]
        members = windows.view(b * m, k * k, d)              # (b*M) x k^2 x d
        query = members.mean(dim=1, keepdim=True)            # (b*M) x 1 x d
        out, _ = self.attn(query, members, members, need_weights=False)
        return self.proj(out.view(b, m, d))


class ResamplerConnector(nn.Module):
    """Flamingo's Perceiver resampler (also IDEFICS2, MiniCPM-V): a fixed set of learned queries
    cross-attends into the patches. Always `n_queries` tokens out, whatever the input size.

    As in Flamingo, the keys and values are the patches concatenated with the latents.
    """

    def __init__(self, d_vis: int, d_model: int, n_queries: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.n_queries = n_queries
        self.latents = nn.Parameter(torch.randn(n_queries, d_vis) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(d_vis), "ln_kv": nn.LayerNorm(d_vis),
                "attn": nn.MultiheadAttention(d_vis, heads, batch_first=True),
                "ln_ff": nn.LayerNorm(d_vis), "ff": mlp(d_vis, d_vis),
            }))
        self.norm = nn.LayerNorm(d_vis)
        self.proj = nn.Linear(d_vis, d_model)

    def out_tokens(self, h: int, w: int) -> int:
        return self.n_queries

    def forward(self, x, grid):
        lat = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        for blk in self.blocks:
            q = blk["ln_q"](lat)
            kv = blk["ln_kv"](torch.cat([x, lat], dim=1))
            lat = lat + blk["attn"](q, kv, kv, need_weights=False)[0]
            lat = lat + blk["ff"](blk["ln_ff"](lat))
        return self.proj(self.norm(lat))


CONNECTORS = {
    "linear": LinearConnector,
    "mlp": MLPConnector,
    "pixel_shuffle": PixelShuffleConnector,
    "avg_pool": AvgPoolConnector,
    "attn_pool": AttnPoolConnector,
    "resampler": ResamplerConnector,
}


def build_connector(name: str, d_vis: int, d_model: int, **kw) -> nn.Module:
    """`name` is a key of CONNECTORS; extra keywords go to the class (s, k, n_queries, ...)."""
    if name not in CONNECTORS:
        raise ValueError(f"unknown connector {name!r}; choose from {sorted(CONNECTORS)}")
    return CONNECTORS[name](d_vis, d_model, **kw)


def n_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
