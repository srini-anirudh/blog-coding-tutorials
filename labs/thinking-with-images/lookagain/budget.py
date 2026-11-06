"""Pixel and token arithmetic from the post's section "What reasoning tokens cannot recover".

Everything here is exact arithmetic. Nothing is measured or modelled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Encoding:
    """One image (or crop) resized for the encoder."""

    region_w: int  # width of the region in original pixels
    region_h: int
    enc_w: int  # width after the resolution policy resized it
    enc_h: int
    patch: int = 14
    merge: int = 1  # 2 means a 2x2 token merge, as in Qwen2-VL

    @property
    def grid(self) -> tuple[int, int]:
        return self.enc_w // self.patch, self.enc_h // self.patch

    @property
    def patches(self) -> int:
        gw, gh = self.grid
        return gw * gh

    @property
    def tokens(self) -> int:
        """Visual tokens the language model sees."""
        return self.patches // (self.merge * self.merge)

    def text_height(self, h_orig: float) -> float:
        """h_enc = h_orig * W_enc / W_region."""
        return h_orig * self.enc_w / self.region_w


def resize_long_side(w: int, h: int, long_side: int) -> tuple[int, int]:
    s = long_side / max(w, h)
    return round(w * s), round(h * s)


def min_width_for_text(h_orig: float, h_needed: float, region_w: int) -> float:
    """Encoder width at which text of height h_orig reaches h_needed pixels."""
    return h_needed * region_w / h_orig


def snap(x: float, multiple: int = 28) -> int:
    """Round up to a multiple, as Qwen2-VL needs sizes divisible by patch * merge = 28."""
    return int(math.ceil(x / multiple) * multiple)


def attention_cost(n_visual: int, t_reason: int) -> int:
    """Attention score count for a trace of T tokens after N_v visual tokens: sum_t (N_v + t) = T*N_v + T(T+1)/2."""
    return t_reason * n_visual + t_reason * (t_reason + 1) // 2


def image_share(n_visual: int, n_text: int) -> float:
    """Fraction of the context that is image tokens."""
    return n_visual / (n_visual + n_text)
