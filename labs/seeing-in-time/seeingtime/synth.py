"""Synthetic videos defined in continuous time, so any sampling rule can be applied to the same scene.

`render_clip` draws the post's ball-and-cup minute. The `*_batch` functions make the small labelled
datasets used by the three training experiments.
"""

from __future__ import annotations

import math

import torch

from .clip import EVENTS

H, W = 48, 72
TABLE_Y, FLOOR_Y = 30, 46
CUP_RGB = (1.0, 0.55, 0.1)  # orange, as in the post
BALL_RGB = (0.2, 0.4, 1.0)


def _box(img, y0, y1, x0, x1, rgb):
    y0, y1 = max(0, int(y0)), min(img.shape[1], int(y1))
    x0, x1 = max(0, int(x0)), min(img.shape[2], int(x1))
    for c in range(3):
        img[c, y0:y1, x0:x1] = rgb[c]


def ball_xy(t: float) -> tuple[float, float]:
    roll, drop = EVENTS[0], EVENTS[1]
    if t < roll.start:
        return 20.0, TABLE_Y - 3
    if t < roll.end:
        return 20.0 + (t - roll.start) / (roll.end - roll.start) * 30.0, TABLE_Y - 3
    if t < drop.end:
        u = (t - drop.start) / (drop.end - drop.start)
        return 50.0 + 4.0 * u, TABLE_Y - 3 + u * u * (FLOOR_Y - TABLE_Y)
    return 54.0, FLOOR_Y - 3


def cup_y(t: float) -> float:
    """Cup base height: on the table, lifted while picked up and drunk from, back down at 52-53 s."""
    lift = 12.0
    if 40 <= t < 44:
        return TABLE_Y - lift * (t - 40) / 4
    if 44 <= t < 52:
        return TABLE_Y - lift
    if 52 <= t < 53:
        return TABLE_Y - lift * (53 - t)
    return TABLE_Y


def render_clip(t: float) -> torch.Tensor:
    """3 x 48 x 72 frame of the clip at time t (seconds). The cup label is dark on its left half only."""
    img = torch.full((3, H, W), 0.92)
    _box(img, TABLE_Y, TABLE_Y + 2, 4, 52, (0.45, 0.3, 0.2))  # table top
    _box(img, FLOOR_Y, H, 0, W, (0.6, 0.6, 0.6))  # floor
    if 38 <= t < 55:  # a person stands at the right
        _box(img, 6, FLOOR_Y, 60, 68, (0.3, 0.6, 0.3))
    cy = cup_y(t)
    _box(img, cy - 8, cy, 8, 16, CUP_RGB)
    img[:, int(cy) - 6:int(cy) - 2, 8:12] = 0.05  # fine detail: the label's left half is dark, right half not
    bx, by = ball_xy(t)
    _box(img, by - 3, by + 3, bx - 3, bx + 3, BALL_RGB)
    return img


def render_times(times) -> torch.Tensor:
    return torch.stack([render_clip(float(t)) for t in times])


# ---- tiny training sets -------------------------------------------------------------------------------

def blob_strip(x: torch.Tensor, width: int = 32, height: int = 6, sigma: float = 1.2) -> torch.Tensor:
    """Frames of a Gaussian blob at sub-pixel positions x (any shape) on a circular track. Out: x.shape + (h, w)."""
    cols = torch.arange(width).float()
    d = (cols - x[..., None]) % width
    d = torch.minimum(d, width - d)
    row = torch.exp(-0.5 * (d / sigma) ** 2)
    return row[..., None, :].expand(*x.shape, height, width).contiguous()


def speed_batch(n: int, durations: tuple[float, float], rule: str, n_frames: int = 8, fps: float = 2.0,
                speeds=(1.0, 2.0, 4.0), g: torch.Generator | None = None):
    """Videos of a blob moving at one of three speeds (pixels per second). Label = speed class.

    rule="uniform": n_frames spread over the whole video. rule="fps": a clip of n_frames at `fps`,
    placed at a random start (Apollo-style). Returns frames (n, n_frames, h, w) and labels (n,).
    """
    y = torch.randint(len(speeds), (n,), generator=g)
    v = torch.tensor(speeds)[y]
    dur = torch.empty(n).uniform_(*durations, generator=g)
    x0 = torch.empty(n).uniform_(0, 32, generator=g)
    k = torch.arange(n_frames).float()
    if rule == "uniform":
        t = (k[None] + 0.5) * dur[:, None] / n_frames
    elif rule == "fps":
        clip = n_frames / fps
        start = torch.rand(n, generator=g) * (dur - clip).clamp(min=0)
        t = start[:, None] + (k[None] + 0.5) / fps
    else:
        raise ValueError(rule)
    return blob_strip(x0[:, None] + v[:, None] * t), y


def event_batch(n: int, fps_choices, g: torch.Generator | None = None, size: int = 8):
    """Videos of 8 to 24 s with one event (a bright square) lasting 1 to 6 s. Label = duration in seconds.

    Each video is sampled at a frame rate drawn from fps_choices. Returns padded frames (n, T, size*size),
    frame times (n, T), a padding mask (n, T) and durations (n,).
    """
    D = torch.empty(n).uniform_(8, 24, generator=g)
    d = torch.empty(n).uniform_(1, 6, generator=g)
    s = torch.rand(n, generator=g) * (D - d)
    f = torch.tensor(fps_choices, dtype=torch.float)[torch.randint(len(fps_choices), (n,), generator=g)]
    T = int(math.ceil(24 * max(fps_choices)))
    k = torch.arange(T).float()
    t = k[None] / f[:, None]  # frames at 0, 1/f, 2/f, ... seconds
    mask = t < D[:, None]
    on = ((t >= s[:, None]) & (t <= (s + d)[:, None]) & mask).float()
    frames = 0.1 * torch.randn(n, T, size * size, generator=g)
    frames[..., : size * size // 2] += on[..., None]  # the event lights up the top half
    frames = frames * mask[..., None]
    return frames, t * mask, mask, d


COLOURS = torch.tensor([[1.0, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.2, 1.0]])


def qa_batch(n: int, bias: float = 0.0, n_frames: int = 8, g: torch.Generator | None = None):
    """Three questions about one short clip, each with its own label.

    colour: which of three colours the ball is (one frame answers it).
    direction: does the ball move left (0) or right (1) (needs two ordered frames).
    order: does the flash at the top come before (1) or after (0) the flash at the bottom (needs order).
    bias in [0, 1]: with this probability the direction is set by the colour (red right, blue left),
    which plants a static shortcut in the data. Frames: (n, T, 3, 12, 12).
    """
    side = 12
    colour = torch.randint(3, (n,), generator=g)
    direction = torch.randint(2, (n,), generator=g)
    forced = torch.rand(n, generator=g) < bias
    direction = torch.where(forced & (colour == 0), torch.ones_like(direction), direction)
    direction = torch.where(forced & (colour == 2), torch.zeros_like(direction), direction)
    order = torch.randint(2, (n,), generator=g)
    x0 = torch.randint(3, 9, (n,), generator=g).float()
    step = 0.6 * (2 * direction.float() - 1)
    k = torch.arange(n_frames).float()
    x = x0[:, None] + step[:, None] * (k[None] - (n_frames - 1) / 2)
    cols = torch.arange(side).float()
    blob = torch.exp(-0.5 * ((cols - x[..., None]) / 0.8) ** 2)  # n, T, side
    frames = torch.zeros(n, n_frames, 3, side, side)
    frames[..., 5:7, :] = blob[:, :, None, None, :] * COLOURS[colour][:, None, :, None, None]
    a = torch.randint(0, n_frames // 2, (n,), generator=g)  # first flash in the first half
    b = torch.randint(n_frames // 2, n_frames, (n,), generator=g)  # second flash in the second half
    top_t = torch.where(order == 1, a, b)
    bot_t = torch.where(order == 1, b, a)
    idx = torch.arange(n)
    frames[idx, top_t, :, 0:2, :] = 1.0
    frames[idx, bot_t, :, 10:12, :] = 1.0
    return frames, {"colour": colour, "direction": direction, "order": order}
