"""Token arithmetic of video: tokens per frame, tokens per video, and what a context window holds."""

from __future__ import annotations


def tokens_per_frame(height: int, width: int, patch: int = 14, merge: int = 2) -> int:
    """tau = (H / p) * (W / p) / r^2. A 448 x 448 frame, 14-pixel patches, 2 x 2 merge -> 256."""
    if height % patch or width % patch:
        raise ValueError("frame size must be a multiple of the patch size")
    gh, gw = height // patch, width // patch
    if gh % merge or gw % merge:
        raise ValueError("patch grid must be a multiple of the merge factor")
    return gh * gw // (merge * merge)


def video_tokens(duration_s: float, fps: float, tau: int) -> int:
    """N = D * f * tau, with the frame count rounded to a whole number of frames."""
    return round(duration_s * fps) * tau


def seconds_that_fit(context: int, fps: float, tau: int) -> float:
    """How many seconds of video fit in `context` visual tokens."""
    return context / (fps * tau)


def attention_vs_ffn_growth(scale: float) -> tuple[float, float]:
    """Multiply the sequence by `scale`: feed-forward cost grows by `scale`, attention scores by `scale ** 2`."""
    return scale, scale ** 2


def slowfast_tokens(slow_frames: int, slow_tokens: int, fast_frames: int, fast_tokens: int) -> int:
    """Total tokens of a slow-fast split, e.g. SlowFast-LLaVA: 10 x (12*24) + 50 x (4*4) = 3,680."""
    return slow_frames * slow_tokens + fast_frames * fast_tokens
