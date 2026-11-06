"""Part VIII of the post: number formats, rounding stagnation and dynamic loss scaling."""

from __future__ import annotations

import torch

FORMATS = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
    "FP8 E4M3": torch.float8_e4m3fn,
    "FP8 E5M2": torch.float8_e5m2,
}


def format_table() -> list[dict]:
    """Bit layout and limits straight from torch.finfo. eps is the spacing just above 1.0."""
    rows = []
    for name, dt in FORMATS.items():
        fi = torch.finfo(dt)
        bits = fi.bits
        mant = round(-torch.log2(torch.tensor(fi.eps)).item())
        rows.append({"format": name, "bits": bits, "exp": bits - 1 - mant, "mant": mant, "max": fi.max,
                     "min_normal": fi.smallest_normal, "eps": fi.eps, "unit_roundoff": fi.eps / 2})
    return rows


def fraction_lost(values: torch.Tensor, dtype: torch.dtype, scale: float = 1.0) -> dict:
    """Cast scale * values to `dtype` and count how many become zero, subnormal, or infinite."""
    x = (values.double() * scale)
    y = x.to(dtype).double()
    fi = torch.finfo(dtype)
    nonzero = x != 0
    n = nonzero.sum().item()
    return {
        "zero": ((y == 0) & nonzero).sum().item() / n,
        "subnormal": ((y != 0) & (y.abs() < fi.smallest_normal)).sum().item() / n,
        "inf": (~torch.isfinite(y)).sum().item() / n,
    }


class DynamicLossScaler:
    """Start high; halve and skip the step on inf/NaN; double after `growth_interval` clean steps."""

    def __init__(self, init_scale: float = 2.0**24, growth_interval: int = 200, factor: float = 2.0) -> None:
        self.scale, self.growth_interval, self.factor = init_scale, growth_interval, factor
        self.clean = 0
        self.skipped = 0

    def update(self, found_inf: bool) -> bool:
        """Record one step. Returns True if the optimizer step should be applied."""
        if found_inf:
            self.scale /= self.factor
            self.clean = 0
            self.skipped += 1
            return False
        self.clean += 1
        if self.clean == self.growth_interval:
            self.scale *= self.factor
            self.clean = 0
        return True
