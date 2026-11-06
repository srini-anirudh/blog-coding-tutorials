"""Reuse the C3 attention tutorial instead of rebuilding what it already has.

C3 (`c03-attention/attnzoo`) builds half-split RoPE, ALiBi and axial 2D RoPE. This lab imports them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_C03 = Path(__file__).resolve().parents[3] / "c03-attention"
if str(_C03) not in sys.path:
    sys.path.insert(0, str(_C03))

from attnzoo.biases import alibi_bias, alibi_slopes  # noqa: E402
from attnzoo.pos2d import Rope2D, sincos_2d_embedding  # noqa: E402
from attnzoo.rope import RotaryEmbedding, rotate, rotate_half  # noqa: E402

__all__ = ["RotaryEmbedding", "Rope2D", "sincos_2d_embedding", "alibi_bias", "alibi_slopes", "rotate", "rotate_half"]
