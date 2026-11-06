"""NaViT's Patch n' Pack, padding masks, and position encodings that do or do not care about the grid shape.

A tiny ViT with 2D rotary position (shape-agnostic, as in Qwen2.5-VL) runs on images of any size. Several
images go into one sequence; a block-diagonal mask keeps each image's patches attending only to themselves.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def rope_2d(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotate the first half of each head's channels by the row index and the second half by the column index.

    x: B x heads x N x d (d divisible by 4). rows, cols: B x N integer grid coordinates.
    """
    d = x.shape[-1]
    quarter = d // 4
    freqs = base ** (-torch.arange(quarter, device=x.device, dtype=torch.float32) / quarter)

    def rotate(part: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        ang = pos[:, None, :, None].float() * freqs  # B x 1 x N x quarter
        cos, sin = ang.cos(), ang.sin()
        a, b = part[..., :quarter], part[..., quarter:]
        return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)

    return torch.cat([rotate(x[..., : d // 2], rows), rotate(x[..., d // 2:], cols)], dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, rows, cols, mask):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rope_2d(q, rows, cols), rope_2d(k, rows, cols)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask[:, None])
        x = x + self.out(att.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    """Patch embedding (a strided conv in spirit, a Linear on flattened patches here) + blocks with 2D RoPE."""

    def __init__(self, patch: int = 8, dim: int = 64, depth: int = 3, heads: int = 4, channels: int = 3):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(patch * patch * channels, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def patches(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """image: C x H x W -> (N x p*p*C patches, row ids, column ids)."""
        c, h, w = image.shape
        p = self.patch
        x = image.reshape(c, h // p, p, w // p, p).permute(1, 3, 2, 4, 0).reshape(-1, p * p * c)
        rows, cols = torch.meshgrid(torch.arange(h // p), torch.arange(w // p), indexing="ij")
        return x, rows.flatten(), cols.flatten()

    def forward(self, x, rows, cols, mask=None):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x, rows, cols, mask)
        return self.norm(x)

    def encode_one(self, image: torch.Tensor) -> torch.Tensor:
        x, r, c = self.patches(image)
        return self(x[None], r[None], c[None])[0]


def pack(model: TinyViT, images: list[torch.Tensor]) -> dict:
    """Concatenate every image's patches into one sequence. segment[i] says which image token i came from."""
    xs, rs, cs, seg = [], [], [], []
    for i, img in enumerate(images):
        x, r, c = model.patches(img)
        xs.append(x), rs.append(r), cs.append(c), seg.append(torch.full((len(x),), i))
    seg = torch.cat(seg)
    return {"x": torch.cat(xs)[None], "rows": torch.cat(rs)[None], "cols": torch.cat(cs)[None],
            "segment": seg, "mask": (seg[:, None] == seg[None, :])[None]}


def pad_batch(model: TinyViT, images: list[torch.Tensor]) -> dict:
    """The usual alternative: one row per image, padded to the longest. valid marks real patches."""
    items = [model.patches(img) for img in images]
    n = max(len(x) for x, _, _ in items)
    dim = items[0][0].shape[1]
    b = len(items)
    x = torch.zeros(b, n, dim)
    rows, cols = torch.zeros(b, n, dtype=torch.long), torch.zeros(b, n, dtype=torch.long)
    valid = torch.zeros(b, n, dtype=torch.bool)
    for i, (xi, ri, ci) in enumerate(items):
        k = len(xi)
        x[i, :k], rows[i, :k], cols[i, :k], valid[i, :k] = xi, ri, ci, True
    # A query may attend to a key only if the key is real. Padded queries still need one allowed key.
    mask = valid[:, None, :].expand(b, n, n).clone()
    return {"x": x, "rows": rows, "cols": cols, "valid": valid, "mask": mask}


def mean_pool(features: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    """B x N x d -> B x d. With valid=None every position, padding included, is averaged."""
    if valid is None:
        return features.mean(1)
    w = valid.float()[..., None]
    return (features * w).sum(1) / w.sum(1)


def first_fit(lengths: list[int], capacity: int) -> list[list[int]]:
    """Greedy packing: longest first, each into the first sequence with room."""
    bins: list[list[int]] = []
    room: list[int] = []
    for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        for j, r in enumerate(room):
            if lengths[i] <= r:
                bins[j].append(i)
                room[j] -= lengths[i]
                break
        else:
            bins.append([i])
            room.append(capacity - lengths[i])
    return bins


def packing_cost(lengths: list[int], batch: int, capacity: int) -> dict:
    """Pad-to-longest batches against first-fit packed sequences of length `capacity`.

    Attention FLOPs are counted per query-key pair actually computed: dense padded attention computes
    B * N_max^2, a block-diagonal (varlen) kernel computes sum n_i^2.
    """
    real = sum(lengths)
    padded_slots, padded_pairs = 0, 0
    for k in range(0, len(lengths), batch):
        chunk = lengths[k: k + batch]
        padded_slots += len(chunk) * max(chunk)
        padded_pairs += len(chunk) * max(chunk) ** 2
    bins = first_fit(lengths, capacity)
    packed_slots = len(bins) * capacity
    return {
        "images": len(lengths), "real_tokens": real,
        "padded_slots": padded_slots, "padded_waste": 1 - real / padded_slots,
        "packed_sequences": len(bins), "packed_slots": packed_slots, "packed_waste": 1 - real / packed_slots,
        "attn_pairs_padded": padded_pairs, "attn_pairs_block_diag": sum(n * n for n in lengths),
    }


def interpolate_pos_embed(pos: torch.Tensor, new_grid: tuple[int, int]) -> torch.Tensor:
    """Learned absolute position embeddings, g x g x d, resized bicubically to a new grid (ViT, CLIP, DINOv2)."""
    g = int(math.isqrt(pos.shape[0]))
    grid = pos.reshape(1, g, g, -1).permute(0, 3, 1, 2)
    out = F.interpolate(grid, size=new_grid, mode="bicubic", align_corners=False)
    return out.permute(0, 2, 3, 1).reshape(new_grid[0] * new_grid[1], -1)
