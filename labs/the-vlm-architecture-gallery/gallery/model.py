"""Assemble encoder -> connector -> fusion -> language model, with every piece swappable."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .connectors import build_connector
from .data import CELL, GRID
from .fusion import FUSIONS, TinyLM


def sincos_2d(h: int, w: int, d: int) -> torch.Tensor:
    """Fixed 2-D sine-cosine position table, (h*w) x d (half the channels for rows, half for columns)."""
    def one_d(n, dd):
        pos = torch.arange(n, dtype=torch.float32)[:, None]
        freq = torch.exp(-math.log(10000.0) * torch.arange(0, dd, 2, dtype=torch.float32) / dd)
        return torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], 1)
    rows = one_d(h, d // 2)[:, None, :].expand(h, w, d // 2)
    cols = one_d(w, d // 2)[None, :, :].expand(h, w, d // 2)
    return torch.cat([rows, cols], -1).reshape(h * w, d)


class PatchEncoder(nn.Module):
    """Stand-in for a frozen vision encoder: a random per-patch MLP plus fixed 2-D positions.

    It is frozen from the start, as most VLMs freeze their encoder in stage 1. It is local (no
    attention across patches), so whatever a connector throws away cannot be recovered upstream.
    """

    def __init__(self, d_vis: int = 64, patch: int = CELL, seed: int = 1234):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.patch = patch
        self.fc1 = nn.Linear(patch * patch, d_vis)
        self.fc2 = nn.Linear(d_vis, d_vis)
        with torch.no_grad():
            for lin in (self.fc1, self.fc2):
                lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) / math.sqrt(lin.in_features))
                lin.bias.zero_()
        self.register_buffer("pos", sincos_2d(GRID, GRID, d_vis), persistent=False)
        self.requires_grad_(False)

    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        """B x 1 x H x W -> B x (H/p * W/p) x p*p, row-major."""
        return F.unfold(img, self.patch, stride=self.patch).transpose(1, 2)

    def forward(self, img):
        x = self.patchify(img)
        return F.layer_norm(self.fc2(F.gelu(self.fc1(x))), (self.fc2.out_features,)) + self.pos


class TinyVLM(nn.Module):
    def __init__(self, lm: TinyLM, connector: str = "mlp", fusion: str = "prefix", d_vis: int = 64,
                 expert_rank: int = 0, xattn_every: int = 1, **connector_kw):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(fusion)
        self.fusion = fusion
        self.lm = lm
        self.grid = (GRID, GRID)
        if fusion == "encfree":
            # Fuyu: raw pixels, one linear layer, straight into the decoder.
            self.encoder = None
            self.connector = build_connector("linear", CELL * CELL, lm.d)
        else:
            self.encoder = PatchEncoder(d_vis)
            self.connector = build_connector(connector, d_vis, lm.d, **connector_kw)
        self.new_lm_modules: list = []
        if fusion == "xattn":
            self.new_lm_modules = lm.add_cross_attention(every=xattn_every)
        elif fusion == "expert":
            self.new_lm_modules = lm.add_visual_experts(rank=expert_rank)

    def visual_tokens(self, img):
        if self.encoder is None:
            x = F.unfold(img, CELL, stride=CELL).transpose(1, 2)
        else:
            x = self.encoder(img)
        return self.connector(x, self.grid)

    def forward(self, img, q_ids):
        """Logits for the last position. img may be None for text-only input."""
        text = self.lm.embed(q_ids)
        if img is None:
            return self.lm.forward_embeds(text)[:, -1]
        vis = self.visual_tokens(img)
        if self.fusion == "xattn":
            return self.lm.forward_embeds(text, mem=vis)[:, -1]
        h = torch.cat([vis, text], 1)
        vis_mask = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
        vis_mask[:, : vis.shape[1]] = True
        return self.lm.forward_embeds(h, vis_mask=vis_mask)[:, -1]

    def lm_sequence_length(self, n_question: int) -> int:
        n_vis = self.connector.out_tokens(*self.grid)
        return n_question if self.fusion == "xattn" else n_vis + n_question
