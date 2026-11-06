"""M-RoPE (Qwen2-VL): one position id becomes three, (t, h, w), each driving a share of the rotation planes.

Position ids
    text   all three axes carry the same 1D counter, so M-RoPE reduces to ordinary RoPE.
    image  t is constant; h and w are the token's row and column; all offset by the running start.
    video  like an image per frame, with t stepping by `time_interval` per temporal patch.
    After a vision block the next start is max(all ids so far) + 1 (the post's bookkeeping rule).

Frequency layouts over the d/2 planes of a head
    chunked      Qwen2-VL: contiguous sections, mrope_section = [16, 24, 24] for a 128-dim head.
    interleaved  Qwen3-VL: t, h, w, t, h, w, ... so every axis gets fast and slow planes.
"""

from __future__ import annotations

import math

import torch

from .c03 import rotate
from .rope import wavelengths


def vision_ids(start: int, t: int, h: int, w: int, time_interval: float = 1.0) -> torch.Tensor:
    """(3, t*h*w) ids for one image (t = 1) or video, in raster order inside each frame."""
    tt = (torch.arange(t) * time_interval).long() + start
    hh = torch.arange(h) + start
    ww = torch.arange(w) + start
    T, H, W = torch.meshgrid(tt, hh, ww, indexing="ij")
    return torch.stack([T, H, W]).reshape(3, -1)


def position_ids(segments: list[tuple], rule: str = "max") -> torch.Tensor:
    """segments: ("text", n) | ("image", h, w) | ("video", t, h, w, time_interval). Returns (3, N).

    rule picks where the next block starts:
      "max"    max(all previous ids) + 1, the correct rule.
      "last"   the previous token's id + 1, read off the t axis as a 1D counter would. The bug the post warns
               about: after an image, t has barely moved, so the text lands inside the image's h and w ids.
      "count"  the number of tokens so far, as if everything were 1D text.
    """
    parts, n_tokens = [], 0
    for seg in segments:
        if not parts:
            start = torch.zeros(3, dtype=torch.long)
        elif rule == "max":
            start = torch.full((3,), int(torch.cat(parts, 1).max()) + 1)
        elif rule == "last":
            start = (parts[-1][0, -1] + 1).expand(3)
        elif rule == "count":
            start = torch.full((3,), n_tokens)
        else:
            raise ValueError(rule)
        kind = seg[0]
        if kind == "text":
            ids = torch.arange(seg[1]).expand(3, -1) + start[:, None]
        elif kind in ("image", "video"):
            t, h, w, dt = (1, seg[1], seg[2], 1.0) if kind == "image" else seg[1:]
            ids = vision_ids(0, t, h, w, dt) + start[:, None]
        else:
            raise ValueError(kind)
        parts.append(ids)
        n_tokens += ids.shape[1]
    return torch.cat(parts, 1)


def axis_of_plane(section: list[int], layout: str) -> torch.Tensor:
    """(d/2,) which axis (0 = t, 1 = h, 2 = w) drives each rotation plane."""
    n = sum(section)
    if layout == "chunked":
        return torch.cat([torch.full((s,), a) for a, s in enumerate(section)])
    if layout == "interleaved":
        axis = torch.zeros(n, dtype=torch.long)
        axis[1 : 3 * section[1] : 3] = 1
        axis[2 : 3 * section[2] : 3] = 2
        return axis
    raise ValueError(layout)


def apply_mrope(x: torch.Tensor, ids: torch.Tensor, freqs: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """x (..., N, d), ids (3, N). Plane i rotates by ids[axis[i]] * freqs[i] (half-split layout)."""
    angles = ids.float().T[:, axis] * freqs.float()[None]  # (N, d/2)
    emb = torch.cat([angles, angles], -1)
    shape = x.shape
    out = rotate(x.reshape(-1, 1, shape[-2], shape[-1]), emb.cos()[None], emb.sin()[None])
    return out.reshape(shape)


def axis_wavelength_range(section: list[int], layout: str, base: float) -> dict[str, tuple[float, float]]:
    """Shortest and longest wavelength (in position-id units) that each axis gets."""
    lam = wavelengths(2 * sum(section), base)
    axis = axis_of_plane(section, layout)
    return {name: (float(lam[axis == a].min()), float(lam[axis == a].max())) for a, name in enumerate("thw")}


def qwen25_time_interval(fps: float, tokens_per_second: float = 2.0, temporal_patch_size: int = 2) -> float:
    """Qwen2.5-VL ties t to seconds: t step per temporal patch = tokens_per_second * patch / fps."""
    return tokens_per_second * temporal_patch_size / fps


def image_footprint(height_px: int, width_px: int, px_per_token: int = 28, v2pe_step: float | None = None) -> dict:
    """How much of the position range one image consumes under each scheme."""
    h, w = math.ceil(height_px / px_per_token), math.ceil(width_px / px_per_token)
    out = {"tokens": h * w, "flat_1d": h * w, "mrope": max(h, w)}
    if v2pe_step is not None:
        out["v2pe"] = h * w * v2pe_step
    return out
