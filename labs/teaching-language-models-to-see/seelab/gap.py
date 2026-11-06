"""The gap between the two models: where projected image vectors sit relative to the language model's embeddings.

Three mismatches from the post, each turned into a number:
    location  cosine of each image token to its nearest row of the embedding table
    scale     norm of image tokens / norm of text embeddings
    content   does the language model write the right caption (measured by the caller)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def text_table(lm, n_real: int) -> torch.Tensor:
    """The rows of the embedding table that belong to real tokens (not padding rows or the new <image> id)."""
    return lm.get_input_embeddings().weight[:n_real].detach().float()


@torch.no_grad()
def nearest_rows(z: torch.Tensor, table: torch.Tensor, k: int = 1, chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k cosine similarity of each row of z (M x d) to the table (V x d)."""
    tn = F.normalize(table, dim=-1)
    vals, idx = [], []
    for i in range(0, z.shape[0], chunk):
        s = F.normalize(z[i:i + chunk].float(), dim=-1) @ tn.T
        v, j = s.topk(k, dim=-1)
        vals.append(v)
        idx.append(j)
    return torch.cat(vals), torch.cat(idx)


@torch.no_grad()
def text_reference(table: torch.Tensor, n: int = 2000, seed: int = 0) -> dict[str, float]:
    """The same statistics for real text embeddings, so the image numbers have something to compare with."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(table.shape[0], generator=g)[:n]
    sims, _ = nearest_rows(table[rows], table, k=2)  # k = 2: the first hit is the row itself
    centroid = table.mean(0)
    return {
        "norm": table.norm(dim=-1).mean().item(),
        "nearest_cos": sims[:, 1].mean().item(),
        "cos_to_text_centroid": F.cosine_similarity(table[rows], centroid[None], dim=-1).mean().item(),
    }


@torch.no_grad()
def image_stats(z: torch.Tensor, table: torch.Tensor) -> dict[str, float]:
    """z: M x d image tokens (after the connector)."""
    sims, _ = nearest_rows(z, table, k=1)
    centroid = table.mean(0)
    return {
        "norm": z.float().norm(dim=-1).mean().item(),
        "nearest_cos": sims[:, 0].mean().item(),
        "cos_to_text_centroid": F.cosine_similarity(z.float(), centroid[None], dim=-1).mean().item(),
    }


def object_token(image, grid: int = 7, background_min: int = 230) -> int:
    """Index (row-major) of the image token whose cell holds the most non-background pixels."""
    a = np.asarray(image.convert("RGB")).astype(int)
    ink = (a.min(-1) < background_min)  # the canvas is light grey, every shape colour has a dark channel
    cell = a.shape[0] // grid
    counts = ink[: cell * grid, : cell * grid].reshape(grid, cell, grid, cell).sum((1, 3))
    return int(counts.argmax())


def object_mask(image, grid: int, background_min: int = 230, min_fraction: float = 0.05) -> torch.Tensor:
    """grid x grid boolean mask of the cells that a shape covers by at least min_fraction."""
    a = np.asarray(image.convert("RGB")).astype(int)
    ink = (a.min(-1) < background_min)
    cell = a.shape[0] // grid
    frac = ink[: cell * grid, : cell * grid].reshape(grid, cell, grid, cell).mean((1, 3))
    return torch.from_numpy(frac >= min_fraction).flatten()
