"""The three scoring designs from "Three ways to score a match", at toy size.

All three share the same two building blocks:
  PageEncoder:  32 x 32 page -> 16 patches -> linear patch embedding -> small Transformer -> 16 x d_model
  QueryEncoder: 4 query words -> embedding -> small Transformer -> 4 x d_model

BiEncoder         s(q, d) = cos(pool(f(q)), pool(g(d)))         one vector per page
LateInteraction   s(q, d) = sum_i max_j <q_i, d_j>              one vector per patch (MaxSim)
CrossEncoder      s(q, d) = h(Transformer([q ; d]))             nothing can be precomputed
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import pages as P


def _encoder(d_model: int, layers: int, heads: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=0.0,
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def patchify(images: torch.Tensor) -> torch.Tensor:
    """B x 3 x 32 x 32 -> B x 16 x 192, patches in row-major order (cell = row * 4 + col)."""
    b = images.shape[0]
    x = images.unfold(2, P.PATCH, P.PATCH).unfold(3, P.PATCH, P.PATCH)  # B 3 4 4 8 8
    return x.permute(0, 2, 3, 1, 4, 5).reshape(b, P.N_PATCHES, 3 * P.PATCH * P.PATCH)


class PageEncoder(nn.Module):
    def __init__(self, d_model: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(3 * P.PATCH * P.PATCH, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(1, P.N_PATCHES, d_model))
        self.body = _encoder(d_model, layers, heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.norm(self.body(self.embed(patchify(images)) + self.pos))


class QueryEncoder(nn.Module):
    def __init__(self, d_model: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(P.VOCAB, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(1, P.QUERY_LEN, d_model))
        self.body = _encoder(d_model, layers, heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.body(self.embed(tokens) + self.pos))


class DualModel(nn.Module):
    """Shared parts of the bi-encoder and late interaction: two towers and a learned temperature."""

    def __init__(self, dim: int, d_model: int = 64, layers: int = 2):
        super().__init__()
        self.dim = dim
        self.pages = PageEncoder(d_model, layers)
        self.queries = QueryEncoder(d_model, layers)
        self.page_proj = nn.Linear(d_model, dim)
        self.query_proj = nn.Linear(d_model, dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # CLIP's initial tau = 0.07

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)


class BiEncoder(DualModel):
    """One vector per page and per query: mean-pool the token states, project, normalise."""

    def encode_pages(self, images: torch.Tensor) -> torch.Tensor:          # B x dim
        return F.normalize(self.page_proj(self.pages(images).mean(1)), dim=-1)

    def encode_queries(self, tokens: torch.Tensor) -> torch.Tensor:        # Q x dim
        return F.normalize(self.query_proj(self.queries(tokens).mean(1)), dim=-1)

    @staticmethod
    def score(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:           # Q x N cosine
        return q @ d.T


def maxsim(q: torch.Tensor, d: torch.Tensor, d_mask: torch.Tensor | None = None) -> torch.Tensor:
    """ColBERT's MaxSim. q: Q x n x dim, d: N x m x dim -> Q x N. For each query token take its best
    page vector, then sum over query tokens. d_mask (N x m, bool) marks real vectors when pages carry
    different numbers of vectors (after pooling or pruning)."""
    sim = torch.einsum("qid,nmd->qnim", q, d)
    if d_mask is not None:
        sim = sim.masked_fill(~d_mask[None, :, None, :], float("-inf"))
    return sim.max(-1).values.sum(-1)


class LateInteraction(DualModel):
    """One vector per patch and per query token, scored with MaxSim."""

    def encode_pages(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:  # B x 16 x dim
        x = self.page_proj(self.pages(images))
        return F.normalize(x, dim=-1) if normalize else x

    def encode_queries(self, tokens: torch.Tensor) -> torch.Tensor:        # Q x 4 x dim
        return F.normalize(self.query_proj(self.queries(tokens)), dim=-1)

    @staticmethod
    def score(q: torch.Tensor, d: torch.Tensor, d_mask: torch.Tensor | None = None) -> torch.Tensor:
        return maxsim(q, d, d_mask)


class CrossEncoder(nn.Module):
    """Query words and page patches in one sequence, a [CLS] slot and a linear head.
    Every query word can attend to every patch in every layer."""

    def __init__(self, d_model: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.patch_embed = nn.Linear(3 * P.PATCH * P.PATCH, d_model)
        self.word_embed = nn.Embedding(P.VOCAB, d_model)
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, d_model))
        self.pos = nn.Parameter(0.02 * torch.randn(1, 1 + P.QUERY_LEN + P.N_PATCHES, d_model))
        self.body = _encoder(d_model, layers, heads)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """tokens: B x 4, images: B x 3 x 32 x 32 (paired row by row) -> B scores."""
        b = tokens.shape[0]
        x = torch.cat([self.cls.expand(b, -1, -1), self.word_embed(tokens),
                       self.patch_embed(patchify(images))], dim=1) + self.pos
        return self.head(self.norm(self.body(x))[:, 0]).squeeze(-1)

    @torch.no_grad()
    def score_pairs(self, tokens: torch.Tensor, images: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Score one query against many pages (tokens: 4, images: N x 3 x 32 x 32) -> N."""
        out = []
        for i in range(0, images.shape[0], chunk):
            im = images[i:i + chunk]
            out.append(self(tokens[None].expand(im.shape[0], -1), im))
        return torch.cat(out)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
