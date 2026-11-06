"""The modality gap and modality bias (section "The modality gap and mixed-modality search").

Derivation from the post: write each image vector as v = v' + g, with g a shared offset. Then
<t, v> = <t, v'> + <t, g>. The second term is the same for every image, so it cannot change the order
of a pool that holds only images. It differs between images and texts, so in a mixed pool it can.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gap_stats(img: torch.Tensor, txt: torch.Tensor) -> dict[str, float]:
    """img, txt: n x d, row i of each is a matching pair. All vectors are L2-normalised first.

    cone_*: mean cosine between two different items of one modality (1 = all in one direction).
    gap: distance between the two modality centroids (Liang et al.'s gap vector norm).
    matched_cos / unmatched_cos: mean cosine of pairs that match and pairs that do not.
    """
    img, txt = F.normalize(img, dim=-1), F.normalize(txt, dim=-1)
    n = img.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)

    def cone(x):
        s = x @ x.T
        return s[off].mean().item()

    cross = img @ txt.T
    return {
        "cone_image": cone(img),
        "cone_text": cone(txt),
        "gap": (img.mean(0) - txt.mean(0)).norm().item(),
        "matched_cos": cross.diag().mean().item(),
        "unmatched_cos": cross[off].mean().item(),
    }


def offset_shift(q: torch.Tensor, cands: torch.Tensor, g: torch.Tensor) -> float:
    """With raw dot products, adding one shared offset g to every candidate adds the same number
    <q, g> to all of a query's scores: <q, v + g> = <q, v> + <q, g>. Returns the largest deviation from
    a constant shift over all queries (zero up to float rounding), so no ranking can change. (Exact for
    dot products; only approximate after renormalisation, as the post says.)"""
    diff = q @ (cands + g).T - q @ cands.T
    return (diff - diff.mean(1, keepdim=True)).abs().max().item()


def offset_preserves_ranking(q: torch.Tensor, cands: torch.Tensor, g: torch.Tensor, tol: float = 1e-4) -> bool:
    return offset_shift(q, cands, g) < tol


def wrong_modality_rate(q: torch.Tensor, img: torch.Tensor, txt_cands: torch.Tensor) -> float:
    """Percent of text queries whose top-1 result in a mixed pool (all images + all text candidates)
    is a text candidate, when the query asked for an image."""
    q, img, txt_cands = (F.normalize(x, dim=-1) for x in (q, img, txt_cands))
    best_img = (q @ img.T).max(1).values
    best_txt = (q @ txt_cands.T).max(1).values
    return 100 * (best_txt > best_img).float().mean().item()


def mixed_pool_top1_accuracy(q: torch.Tensor, img: torch.Tensor, txt_cands: torch.Tensor) -> float:
    """Percent of queries i whose top-1 in the mixed pool is image i (the image they describe)."""
    q, img, txt_cands = (F.normalize(x, dim=-1) for x in (q, img, txt_cands))
    s = torch.cat([q @ img.T, q @ txt_cands.T], dim=1)
    return 100 * (s.argmax(1) == torch.arange(q.shape[0])).float().mean().item()


def image_pool_top1_accuracy(q: torch.Tensor, img: torch.Tensor) -> float:
    q, img = F.normalize(q, dim=-1), F.normalize(img, dim=-1)
    return 100 * ((q @ img.T).argmax(1) == torch.arange(q.shape[0])).float().mean().item()


def center(x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Remove a modality's mean direction and renormalise: one simple way to close the gap."""
    return F.normalize(F.normalize(x, dim=-1) - mean, dim=-1)
