"""The one-minute ball-and-cup clip from the post, and the frame-selection rules applied to it.

Event times match the post's Figure 3. Sample times match the figure's code: uniform sampling takes the
midpoint of each of N equal slices, and fixed-rate sampling takes the midpoint of each 1/f-second slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

CLIP_SECONDS = 60.0


@dataclass(frozen=True)
class Event:
    name: str
    start: float
    end: float


EVENTS = (
    Event("ball rolls", 8.0, 20.0),
    Event("ball drops", 20.0, 21.5),
    Event("cup picked up", 40.0, 44.0),
    Event("person drinks", 44.0, 50.0),
    Event("cup put down", 52.0, 53.0),
)


def uniform_times(n: int, duration: float = CLIP_SECONDS) -> list[float]:
    """N frames, evenly spaced: the midpoint of each of N equal slices."""
    return [(i + 0.5) * duration / n for i in range(n)]


def fps_times(fps: float, duration: float = CLIP_SECONDS) -> list[float]:
    """A fixed frame rate: one frame at the midpoint of every 1/fps-second slice."""
    dt = 1.0 / fps
    out, j = [], 0
    while j * dt + dt / 2 < duration:
        out.append(j * dt + dt / 2)
        j += 1
    return out


def frames_in(event: Event, times: list[float]) -> int:
    return sum(event.start <= t <= event.end for t in times)


def coverage(times: list[float], events=EVENTS) -> dict:
    """What the post's Figure 3 reads out: events seen, and whether the roll direction is recoverable."""
    seen = [e.name for e in events if frames_in(e, times) > 0]
    roll = frames_in(events[0], times)
    return {"frames": len(times), "events_seen": len(seen), "seen": seen, "roll_frames": roll,
            "direction_recoverable": roll >= 2}


def apollo_clip_times(duration: float, fps: float, frames_per_clip: int, max_frames: int) -> list[float]:
    """Apollo-style sampling: short clips at a constant rate; long videos space out the clips, not the frames.

    Inside every clip the gap between frames is exactly 1 / fps, whatever the video length.
    """
    clip_len = frames_per_clip / fps
    n_clips = max(1, min(max_frames // frames_per_clip, int(duration // clip_len)))
    starts = [i * (duration - clip_len) / max(1, n_clips - 1) for i in range(n_clips)] if n_clips > 1 else [0.0]
    return [s + k / fps for s in starts for k in range(frames_per_clip)]


def keyframes(features: torch.Tensor, threshold: float) -> list[int]:
    """Keep a frame only if its feature differs from the last kept frame (cosine similarity below threshold).

    A teaching version of the LongVU idea; LongVU itself uses DINOv2 features and drops frames in windows.
    """
    f = torch.nn.functional.normalize(features.flatten(1).float(), dim=1)
    kept = [0]
    for i in range(1, f.shape[0]):
        if float(f[i] @ f[kept[-1]]) < threshold:
            kept.append(i)
    return kept


def topk_relevance(relevance: torch.Tensor, k: int) -> list[int]:
    """Query-aware selection with no coverage term: the k most relevant frames."""
    return sorted(relevance.topk(k).indices.tolist())


def relevance_with_coverage(relevance: torch.Tensor, k: int, bins: int) -> list[int]:
    """Relevance plus coverage, in the spirit of Adaptive Keyframe Sampling (not its exact algorithm).

    Split the video into `bins` equal segments, give each segment a share of k in proportion to its
    total relevance (at least one frame each), and take the most relevant frames inside each segment.
    """
    n = relevance.shape[0]
    edges = [round(i * n / bins) for i in range(bins + 1)]
    mass = torch.stack([relevance[edges[i]:edges[i + 1]].sum() for i in range(bins)])
    share = torch.ones(bins, dtype=torch.long)
    extra = k - bins
    if extra > 0:
        share += torch.floor(mass / mass.sum() * extra).long()
        for i in mass.argsort(descending=True)[: k - int(share.sum())]:
            share[i] += 1
    out = []
    for i in range(bins):
        seg = relevance[edges[i]:edges[i + 1]]
        take = min(int(share[i]), seg.shape[0])
        out += (seg.topk(take).indices + edges[i]).tolist()
    return sorted(out)
