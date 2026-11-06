"""Position ideas from the post's "Architectural innovations": M-RoPE, absolute-time video, interleaving, V2PE.

C7 (`tinyvlm.positions.position_ids_3d`) builds Qwen2-VL's 3D positions for one image. This module adds what the
families changed afterwards:

    mrope_ids         text, images and videos in one sequence (Qwen2-VL), with an optional time interval per
                      video so that frame positions follow seconds (Qwen2.5-VL)
    axis_per_freq     which rotary frequencies carry time, height and width: in contiguous blocks (Qwen2-VL,
                      Qwen2.5-VL) or interleaved (Qwen3-VL)
    v2pe_positions    InternVL3's variable visual position encoding: visual tokens advance the position by
                      delta < 1, so a long image sequence uses less of the context's position range
"""

from __future__ import annotations

import torch


def mrope_ids(segments: list, merge: int = 2) -> torch.Tensor:
    """3 x L position ids for a sequence of segments.

    A segment is ("text", n), ("image", (t, h, w)) or ("video", (t, h, w), interval). (t, h, w) is the patch grid
    before the merge, as Qwen's processors report it in image_grid_thw. Text advances all three axes by 1 per token.
    A vision block starts at the current position; height and width count merged rows and columns; time advances
    by `interval` per temporal patch (1 for Qwen2-VL; tokens_per_second * seconds_per_patch for Qwen2.5-VL).
    The next segment starts at the current position + max(h, w) / merge.
    """
    out, pos = [], 0
    for seg in segments:
        kind = seg[0]
        if kind == "text":
            n = seg[1]
            out.append(torch.arange(pos, pos + n).expand(3, -1))
            pos += n
            continue
        t, h, w = seg[1]
        interval = seg[2] if kind == "video" and len(seg) > 2 else 1
        h, w = h // merge, w // merge
        tt = (torch.arange(t) * interval).long() + pos
        hh, ww = torch.arange(h) + pos, torch.arange(w) + pos
        grid = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij")).reshape(3, -1)
        out.append(grid)
        pos += max(h, w)
    return torch.cat(out, dim=1)


def positions_used(ids: torch.Tensor) -> int:
    """How far into the position range the sequence reaches (max id + 1)."""
    return int(ids.max()) + 1


def axis_per_freq(section: list[int], interleaved: bool) -> list[int]:
    """For each of the head_dim / 2 rotary frequencies (fastest first), the axis it rotates with: 0 = T, 1 = H, 2 = W.

    Contiguous (Qwen2-VL, Qwen2.5-VL): the first section[0] frequencies are T, the next section[1] H, then W.
    So time only gets the fast frequencies and width only the slow ones.
    Interleaved (Qwen3-VL): T, H, W, T, H, W, ... for the first 3 * section[1] frequencies, then T for the rest.
    Each axis now spans fast and slow frequencies.
    """
    n = sum(section)
    if not interleaved:
        return [0] * section[0] + [1] * section[1] + [2] * section[2]
    axes = [0] * n
    for axis in (1, 2):
        for i in range(axis, 3 * section[axis], 3):
            axes[i] = axis
    return axes


def v2pe_positions(segments: list, delta: float) -> torch.Tensor:
    """1D positions where text advances by 1 and each visual token by `delta` (InternVL3's V2PE).

    segments: ("text", n) or ("image", n_tokens). delta = 1 is ordinary 1D RoPE.
    """
    out, pos = [], 0.0
    for kind, n in segments:
        step = 1.0 if kind == "text" else delta
        out.append(pos + step * torch.arange(n, dtype=torch.float64))
        pos += step * n
    return torch.cat(out)
