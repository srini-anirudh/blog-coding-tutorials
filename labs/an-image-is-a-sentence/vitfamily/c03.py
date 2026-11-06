"""Reuse the C3 attention tutorial instead of rebuilding Swin's window machinery.

C3 (c03-attention/attnzoo) already builds and tests patchify, window_partition, window_merge,
shifted_window_mask and WindowAttention. This lab imports them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_C03 = Path(__file__).resolve().parents[3] / "c03-attention"
if str(_C03) not in sys.path:
    sys.path.insert(0, str(_C03))

from attnzoo.device import be_polite, hardware_note  # noqa: E402
from attnzoo.vision import (  # noqa: E402
    WindowAttention,
    patchify,
    shifted_window_mask,
    window_merge,
    window_partition,
)

__all__ = [
    "WindowAttention",
    "be_polite",
    "hardware_note",
    "patchify",
    "shifted_window_mask",
    "window_merge",
    "window_partition",
]
