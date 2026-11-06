"""NF4, blockwise absmax and double quantization, written from the QLoRA paper (Dettmers et al. 2023).

A 4-bit format is a codebook of 16 values plus a rule that maps each weight to its nearest codeword.
NF4 puts the codewords at quantiles of a standard normal, so each bin gets equal probability mass.
Weights are first divided by their block's absmax, so they land in [-1, 1] where the codebook lives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# The NF4 codebook as published with QLoRA (bitsandbytes `NF4` map, also listed in the paper's appendix).
NF4_PUBLISHED = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0])


def nf4_codebook(offset: float = 0.9677083) -> torch.Tensor:
    """Build NF4 from normal quantiles. 8 positive levels, 7 negative levels, and an exact zero.

    The two halves are built separately so that 0 is a codeword (the asymmetry the post describes).
    `offset` is the outermost quantile used, the value bitsandbytes uses; it keeps the tails finite.
    """
    icdf = torch.special.ndtri
    pos = icdf(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])   # 8 values > 0
    neg = -icdf(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])  # 7 values < 0
    v = torch.cat([pos, torch.zeros(1, dtype=torch.float64), neg]).sort().values
    return (v / v.abs().max()).float()


def uniform_codebook(levels: int = 16) -> torch.Tensor:
    """Evenly spaced codewords over [-1, 1]: the integer-style alternative. With 16 levels, 0 is not a codeword."""
    return torch.linspace(-1, 1, levels)


def quantize_blockwise(w: torch.Tensor, codebook: torch.Tensor, block: int = 64):
    """Return (codes uint8, absmax fp32 per block). block=None means one absmax for the whole tensor."""
    flat = w.detach().float().reshape(-1)
    block = block or flat.numel()
    assert flat.numel() % block == 0, "tensor size must be a multiple of the block size"
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    normed = blocks / absmax[:, None]
    # Nearest codeword. The codebook is sorted, so bucketize on the midpoints is exact and cheap.
    cb = codebook.to(normed.device)
    mids = (cb[1:] + cb[:-1]) / 2
    codes = torch.bucketize(normed, mids).to(torch.uint8)
    return codes, absmax


def dequantize_blockwise(codes: torch.Tensor, absmax: torch.Tensor, codebook: torch.Tensor, shape) -> torch.Tensor:
    return (codebook.to(absmax.device)[codes.long()] * absmax[:, None]).view(shape)


def double_quantize(absmax: torch.Tensor, block: int = 256):
    """Quantize the fp32 absmax constants to int8 in blocks of 256 (QLoRA's double quantization).

    The constants are all positive, so they are centred by their mean first. QLoRA uses an 8-bit float format;
    this uses symmetric int8 with one fp32 scale per block, which costs the same number of bits.
    """
    mean = absmax.mean()
    c = absmax - mean
    pad = (-c.numel()) % block
    c = F.pad(c, (0, pad)).view(-1, block)
    scale = c.abs().amax(dim=1).clamp_min(1e-12) / 127
    q = torch.round(c / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale, mean, absmax.numel()


def double_dequantize(q, scale, mean, n) -> torch.Tensor:
    return (q.float() * scale[:, None]).view(-1)[:n] + mean


def storage_bits_per_param(n: int, block: int = 64, double_quant: bool = True, dq_block: int = 256) -> float:
    """Bytes actually held, counted from tensor sizes: 4-bit codes packed two per byte, plus the constants."""
    n_blocks = n // block
    code_bytes = n / 2
    if double_quant:
        n_dq = -(-n_blocks // dq_block)
        const_bytes = n_blocks * 1 + n_dq * 4 + 4  # int8 absmax, fp32 scale per 256 blocks, one fp32 mean
    else:
        const_bytes = n_blocks * 4
    return 8 * (code_bytes + const_bytes) / n


class NF4Linear(nn.Module):
    """A frozen linear layer stored as NF4 codes. The forward pass dequantizes W, uses it once and drops it.

    The base receives no gradient. Gradients still flow through the dequantized W to the layer's input,
    which is what lets a LoRA branch on top learn (QLoRA's gradient path).
    """

    def __init__(self, base: nn.Linear, block: int = 64, double_quant: bool = True):
        super().__init__()
        self.shape, self.block, self.double_quant = tuple(base.weight.shape), block, double_quant
        self.in_features, self.out_features = base.in_features, base.out_features
        self.compute_dtype = base.weight.dtype
        codes, absmax = quantize_blockwise(base.weight, NF4_PUBLISHED, block)
        self.register_buffer("codes", codes)
        if double_quant:
            q, scale, mean, n = double_quantize(absmax)
            self.register_buffer("absmax_q", q)
            self.register_buffer("absmax_scale", scale)
            self.register_buffer("absmax_mean", mean)
            self._n_blocks = n
        else:
            self.register_buffer("absmax", absmax)
        self.register_buffer("codebook", NF4_PUBLISHED.clone())
        self.bias = None if base.bias is None else nn.Parameter(base.bias.detach().clone(), requires_grad=False)

    @property
    def weight(self) -> torch.Tensor:
        absmax = (double_dequantize(self.absmax_q, self.absmax_scale, self.absmax_mean, self._n_blocks)
                  if self.double_quant else self.absmax)
        return dequantize_blockwise(self.codes, absmax, self.codebook, self.shape).to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias)


def quantize_model_linears(model: nn.Module, targets, **kw) -> int:
    """Swap each target nn.Linear for an NF4Linear, in place. Returns the number swapped."""
    count = 0
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, NF4Linear(child, **kw))
                count += 1
    return count
