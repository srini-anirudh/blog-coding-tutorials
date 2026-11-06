"""Block-wise 8-bit storage for optimizer state.

Split a tensor into blocks, divide each block by its own absolute max, and store the index of the
nearest value in a 256-entry code table. Two tables:
  linear : evenly spaced, like plain int8.
  log    : logarithmically spaced, so small magnitudes keep relative resolution. This is a simple
           stand-in for the dynamic-exponent code of Dettmers et al. (2021), not the same table.
"""

from __future__ import annotations

import torch


def code_table(kind: str, signed: bool) -> torch.Tensor:
    if kind == "linear":
        t = torch.linspace(-1, 1, 255) if signed else torch.linspace(0, 1, 256)
    elif kind == "log":
        if signed:  # 0 and +-2^(-16 k / 126): 255 values spanning 16 octaves
            mag = 2.0 ** (-16 * torch.arange(127) / 126)
            t = torch.cat([-mag, torch.zeros(1), mag])
        else:  # 0 and 2^(-24 k / 254): 256 values spanning 24 octaves (v holds squares)
            t = torch.cat([torch.zeros(1), 2.0 ** (-24 * torch.arange(255) / 254)])
    else:
        raise ValueError(kind)
    return t.sort().values


class BlockQuantized:
    def __init__(self, x: torch.Tensor, block: int | None = 2048, kind: str = "log", signed: bool = True):
        self.shape, self.dtype, self.numel = x.shape, x.dtype, x.numel()
        self.block = block or self.numel
        self.table = code_table(kind, signed)
        flat = x.detach().reshape(-1).float()
        pad = (-self.numel) % self.block
        blocks = torch.nn.functional.pad(flat, (0, pad)).view(-1, self.block)
        self.scale = blocks.abs().amax(1, keepdim=True).clamp_min(1e-30)  # one fp32 per block
        mids = (self.table[1:] + self.table[:-1]) / 2
        self.codes = torch.bucketize(blocks / self.scale, mids).to(torch.uint8)  # nearest entry

    def dequantize(self) -> torch.Tensor:
        x = self.table[self.codes.long()] * self.scale
        return x.reshape(-1)[: self.numel].reshape(self.shape).to(self.dtype)

    def nbytes(self) -> int:
        """1 byte per value plus 4 bytes per block scale (the shared code table is not counted)."""
        return self.codes.numel() + 4 * self.scale.numel()
