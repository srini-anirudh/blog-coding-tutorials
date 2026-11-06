"""A small two-tower VLM: ViT encoder -> m x m merge -> MLP connector -> causal decoder LLM.

Attention is written with explicit matmuls so a FLOP counter can tell attention-score FLOPs
(`bmm`) apart from weight FLOPs (`mm` / `addmm`). Shapes, with B images, P patches, V tokens:

    pixels      B x 3 x H x W
    patchify    B x P x (3 p p)      -> patch embed (Linear)   B x P x d_v
    encoder     B x P x d_v
    merge m     B x (P / m^2) x (d_v m^2)     (pixel shuffle; or avg-pool to `pool_tokens`)
    connector   B x V x d_l
    LLM input   [visual tokens ; text tokens]  B x (V + T) x d_l  -> logits over the vocabulary
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TwoTowerConfig:
    image_size: int = 32
    patch_size: int = 4
    v_dim: int = 64
    v_layers: int = 2
    v_heads: int = 4
    merge: int = 1
    pool_tokens: int | None = None  # if set, average-pool the patch grid to this many tokens instead of merging
    l_dim: int = 64
    l_layers: int = 2
    l_heads: int = 4
    vocab: int = 64
    max_text: int = 16
    mlp_ratio: int = 4

    @property
    def patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def visual_tokens(self) -> int:
        if self.pool_tokens is not None:
            return self.pool_tokens
        return self.patches // (self.merge * self.merge)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int, causal: bool):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, mlp_ratio * dim)
        self.fc2 = nn.Linear(mlp_ratio * dim, dim)

    def attend(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // self.heads)  # B x h x s x s
        if self.causal:
            mask = torch.ones(s, s, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        y = scores.softmax(-1) @ v  # B x h x s x d_h
        return self.out(y.transpose(1, 2).reshape(b, s, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attend(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


def patchify(pixels: torch.Tensor, p: int) -> torch.Tensor:
    b, c, h, w = pixels.shape
    x = pixels.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // p) * (w // p), c * p * p)


def pixel_shuffle(x: torch.Tensor, m: int) -> torch.Tensor:
    """Fold each m x m block of the patch grid into one token: B x P x d -> B x (P/m^2) x (d m^2).

    Same result as `pixel_shuffle` in C7 (`c07-vlm/tinyvlm/projector.py`); a test checks this.
    """
    if m == 1:
        return x
    b, n, d = x.shape
    s = math.isqrt(n)
    x = x.reshape(b, s, s // m, d * m).permute(0, 2, 1, 3)
    x = x.reshape(b, s // m, s // m, d * m * m).permute(0, 2, 1, 3)
    return x.reshape(b, (s // m) ** 2, d * m * m)


def pool_grid(x: torch.Tensor, tokens: int) -> torch.Tensor:
    """Average-pool the square patch grid down to `tokens` (a square number) tokens."""
    b, n, d = x.shape
    s, t = math.isqrt(n), math.isqrt(tokens)
    grid = x.transpose(1, 2).reshape(b, d, s, s)
    return F.adaptive_avg_pool2d(grid, t).reshape(b, d, t * t).transpose(1, 2)


class VisionTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(3 * cfg.patch_size**2, cfg.v_dim)
        self.pos = nn.Parameter(torch.randn(1, cfg.patches, cfg.v_dim) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg.v_dim, cfg.v_heads, cfg.mlp_ratio, causal=False) for _ in range(cfg.v_layers))
        self.ln = nn.LayerNorm(cfg.v_dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(patchify(pixels, self.cfg.patch_size)) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.ln(x)


class Connector(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.v_dim if cfg.pool_tokens is not None else cfg.v_dim * cfg.merge**2
        self.fc1 = nn.Linear(d_in, cfg.l_dim)
        self.fc2 = nn.Linear(cfg.l_dim, cfg.l_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.cfg.pool_tokens is not None:
            feats = pool_grid(feats, self.cfg.pool_tokens)
        else:
            feats = pixel_shuffle(feats, self.cfg.merge)
        return self.fc2(F.gelu(self.fc1(feats)))


class LanguageTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab, cfg.l_dim)
        self.pos = nn.Parameter(torch.randn(1, cfg.visual_tokens + cfg.max_text, cfg.l_dim) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg.l_dim, cfg.l_heads, cfg.mlp_ratio, causal=True) for _ in range(cfg.l_layers))
        self.ln = nn.LayerNorm(cfg.l_dim)
        self.head = nn.Linear(cfg.l_dim, cfg.vocab, bias=False)

    def forward(self, visual: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        x = torch.cat([visual, self.embed(text_ids)], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


class TwoTowerVLM(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionTower(cfg)
        self.connector = Connector(cfg)
        self.llm = LanguageTower(cfg)

    def forward(self, pixels: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        """Logits for the text positions only: B x T x vocab."""
        logits = self.llm(self.connector(self.vision(pixels)), text_ids)
        return logits[:, self.cfg.visual_tokens :]

    def loss(self, pixels: torch.Tensor, text_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Next-token loss; `targets` uses -100 for positions that are not supervised."""
        logits = self(pixels, text_ids)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


def matmul_params(module: nn.Module) -> int:
    """Parameters that take part in a weight matmul: the weights of every nn.Linear (no biases)."""
    return sum(m.weight.numel() for m in module.modules() if isinstance(m, nn.Linear))


def set_trainable(model: TwoTowerVLM, vision: bool, connector: bool, llm: bool) -> None:
    for part, flag in ((model.vision, vision), (model.connector, connector), (model.llm, llm)):
        for p in part.parameters():
            p.requires_grad_(flag)
