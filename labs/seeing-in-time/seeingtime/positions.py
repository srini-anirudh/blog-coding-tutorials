"""Telling the model when: temporal position ids, multimodal RoPE, and its frequency layouts.

Four ways to mark time are built here:
  index    - the temporal id goes up by one per frame group (Qwen2-VL's M-RoPE)
  abstime  - the temporal id follows real seconds (Qwen2.5-VL: tokens_per_second * seconds per group)
  text     - a timestamp written as text before each frame group, positions stay compact (Qwen3-VL)
  v2pe     - visual tokens advance the 1D position by a fraction delta < 1 (InternVL3's V2PE)
The image-only version of these ids is in c07-vlm/tinyvlm/positions.py.
"""

from __future__ import annotations

import torch


def vision_position_ids(start: int, grid_t: int, grid_h: int, grid_w: int, time_interval: float = 1.0):
    """3 x (t*h*w) ids for one video block, ordered t, then row, then column.

    Same arithmetic as Qwen2.5-VL's get_vision_position_ids: the temporal id is floor(k * interval)
    plus the start, height and width ids are the grid cell plus the start.
    """
    t = (torch.arange(grid_t) * time_interval).long() + start
    h = torch.arange(grid_h) + start
    w = torch.arange(grid_w) + start
    T, Hh, Ww = torch.meshgrid(t, h, w, indexing="ij")
    return torch.stack([T, Hh, Ww]).reshape(3, -1)


def temporal_interval(mode: str, fps: float, temporal_patch: int = 2, tokens_per_second: float = 2.0) -> float:
    """Gap between the temporal ids of consecutive frame groups.

    index: 1 for every video. abstime: tokens_per_second * (temporal_patch / fps) seconds per group,
    so equal seconds give equal gaps (Qwen2.5-VL ships tokens_per_second = 2).
    """
    if mode == "index":
        return 1.0
    if mode == "abstime":
        return tokens_per_second * temporal_patch / fps
    raise ValueError(mode)


def text_then_video_then_text(n_before: int, grid: tuple[int, int, int], n_after: int, interval: float):
    """A full sequence: text tokens get (p, p, p); video gets 3D ids; text resumes after the largest id."""
    before = torch.arange(n_before).expand(3, -1)
    vid = vision_position_ids(n_before, *grid, time_interval=interval)
    nxt = int(vid.max()) + 1
    after = (torch.arange(n_after) + nxt).expand(3, -1)
    return torch.cat([before, vid, after], dim=1)


def timestamp_text(seconds: float, style: str = "seconds") -> str:
    """Qwen3-VL-style timestamps: '<3.0 seconds>' or hours:minutes:seconds."""
    if style == "seconds":
        return f"<{seconds:.1f} seconds>"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"<{int(h):02d}:{int(m):02d}:{s:04.1f}>"


def v2pe_positions(is_visual: torch.Tensor, delta: float) -> torch.Tensor:
    """V2PE: +1 for each text token, +delta for each visual token (float positions)."""
    steps = torch.where(is_visual, torch.tensor(delta), torch.tensor(1.0))
    return torch.cat([torch.zeros(1), steps[:-1]]).cumsum(0)


# ---- multimodal RoPE -----------------------------------------------------------------------------------

def frequency_axes(head_dim: int, sections: tuple[int, int, int], interleaved: bool) -> torch.Tensor:
    """For each of the head_dim/2 rotary frequencies, which axis drives it: 0 = time, 1 = height, 2 = width.

    Contiguous (Qwen2-VL, sections [16, 24, 24]): the first 16 frequencies, the fastest, go to time.
    Interleaved (Qwen3-VL, sections [24, 20, 20]): t, h, w, t, h, w, ... so every axis spans the spectrum.
    """
    half = head_dim // 2
    if sum(sections) != half:
        raise ValueError("sections must sum to head_dim / 2")
    if not interleaved:
        return torch.cat([torch.full((n,), a) for a, n in enumerate(sections)])
    axes = torch.zeros(half, dtype=torch.long)
    for a in (1, 2):
        axes[a: 3 * sections[a]: 3] = a
    return axes


def inv_frequencies(head_dim: int, theta: float) -> torch.Tensor:
    return 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)


def mrope_angles(pos3: torch.Tensor, axes: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """pos3 (3, N) ids -> (N, head_dim/2) angles: frequency i rotates by the id of axis axes[i]."""
    return pos3[axes].T.float() * inv_freq[None]


def rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE: x (..., N, head_dim), angles (N, head_dim/2)."""
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def axis_wavelengths(axes: torch.Tensor, inv_freq: torch.Tensor) -> dict[str, tuple[float, float]]:
    """Shortest and longest wavelength (in position ids) that each axis owns."""
    out = {}
    for a, name in enumerate(("time", "height", "width")):
        lam = 2 * torch.pi / inv_freq[axes == a]
        out[name] = (float(lam.min()), float(lam.max()))
    return out
