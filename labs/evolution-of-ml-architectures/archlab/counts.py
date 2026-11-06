"""Parameters, bytes and FLOPs from a config alone, plus real modules to check the formulas against.

The formulas follow the post's sizing section. Every one is checked in the tests by building the
layer in PyTorch and counting `p.numel()`.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------- one layer of each family, width d

def family_formulas(d: int, k: int = 3, E: int = 2, s: int = 16) -> dict[str, int]:
    """Closed-form parameters per layer at a common width d, biases included where the post includes them."""
    return {
        "MLP": d * d + d,
        "Conv 3x3": k * k * d * d + d,
        "Depthwise sep. 3x3": (k * k * d + d) + (d * d + d),
        "Elman RNN": 2 * d * d + d,
        "GRU": 3 * (2 * d * d + d),
        "LSTM": 4 * (2 * d * d + d),
        "Transformer": 4 * d * d + 8 * d * d + 4 * d,  # attention 4d^2, GELU MLP 8d^2, two norms
        "Mamba (E=2)": mamba_block_params(d, E=E, s=s),
    }


def mamba_block_params(d: int, E: int = 2, s: int = 16, conv: int = 4, dt_rank: int | None = None) -> int:
    """Mamba-1 mixer: in_proj d->2Ed, depthwise conv, x_proj Ed->(r+2s), dt_proj r->Ed, A_log, D, out_proj."""
    e = E * d
    r = dt_rank or math.ceil(d / 16)
    return 2 * e * d + (e * conv + e) + e * (r + 2 * s) + (r * e + e) + e * s + e + e * d


class Elman(nn.Module):
    """h_t = tanh(W_h h_{t-1} + W_x x_t + b): one bias, as in the post's formula."""

    def __init__(self, d: int):
        super().__init__()
        self.wx = nn.Linear(d, d)
        self.wh = nn.Linear(d, d, bias=False)


class TransformerLayer(nn.Module):
    """Pre-norm block with bias-free projections and two LayerNorms: 12 d^2 + 4d, as the post counts it."""

    def __init__(self, d: int, d_ff: int | None = None, swiglu: bool = False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkvo = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(4))
        d_ff = d_ff or (round(8 * d / 3 / 128) * 128 if swiglu else 4 * d)
        n_mats = 3 if swiglu else 2
        self.mlp = nn.ModuleList(
            nn.Linear(d, d_ff, bias=False) if i < n_mats - 1 else nn.Linear(d_ff, d, bias=False) for i in range(n_mats)
        )


def family_modules(d: int) -> dict[str, nn.Module]:
    """The same layers as `family_formulas`, built from torch modules (on the meta device: no memory)."""
    with torch.device("meta"):
        return {
            "MLP": nn.Linear(d, d),
            "Conv 3x3": nn.Conv2d(d, d, 3),
            "Depthwise sep. 3x3": nn.Sequential(nn.Conv2d(d, d, 3, groups=d), nn.Conv2d(d, d, 1)),
            "Elman RNN": Elman(d),
            "GRU": nn.GRU(d, d, bias=True),
            "LSTM": nn.LSTM(d, d, bias=True),
            "Transformer": TransformerLayer(d),
        }


# ---------------------------------------------------------------- a decoder-only Transformer from its config

def decoder_params(L: int, d: int, d_ff: int, n_h: int, n_kv: int, d_h: int, V: int, tied: bool = False) -> dict:
    """The post's worked example: attention 2 d n_h d_h + 2 d n_kv d_h, SwiGLU 3 d d_ff, RMSNorm 2d per layer."""
    attn = 2 * d * n_h * d_h + 2 * d * n_kv * d_h
    mlp = 3 * d * d_ff
    norms = 2 * d
    emb = V * d * (1 if tied else 2)
    total = emb + L * (attn + mlp + norms) + d  # + final norm
    return {"embeddings": emb, "attention/layer": attn, "mlp/layer": mlp, "per layer": attn + mlp + norms,
            "total": total}


LLAMA3_8B = dict(L=32, d=4096, d_ff=14336, n_h=32, n_kv=8, d_h=128, V=128256)
LLAMA3_70B = dict(L=80, d=8192, d_ff=28672, n_h=64, n_kv=8, d_h=128, V=128256)


def kv_cache_bytes(L: int, n_ctx: int, n_kv: int, d_h: int, bytes_per: int = 2, batch: int = 1) -> int:
    """K and V for every layer, KV head and position: 2 L N n_kv d_h bytes."""
    return 2 * L * n_ctx * n_kv * d_h * bytes_per * batch


def training_bytes(n: int) -> int:
    """Mixed-precision Adam: bf16 weights 2 + bf16 grads 2 + fp32 master 4 + Adam m 4 + v 4 = 16 bytes."""
    return 16 * n


def attention_to_param_flops(n_ctx: int, d: int) -> float:
    """Per token per layer: 4 N d for QK^T and AV against 2 x 12 d^2 for the matmuls. Ratio N / (6 d)."""
    return (4 * n_ctx * d) / (2 * 12 * d * d)


# ---------------------------------------------------------------- the CNN history in parameters

def lenet5() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2), nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
        nn.Flatten(), nn.Linear(400, 120), nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10))


def alexnet() -> nn.Sequential:
    """The 2012 layer widths (96-256-384-384-256), without the two-GPU grouping."""
    f = nn.Sequential(
        nn.Conv2d(3, 96, 11, stride=4, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(96, 256, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(), nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2))
    c = nn.Sequential(nn.Flatten(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(),
                      nn.Linear(4096, 1000))
    return nn.Sequential(f, c)


def vgg16() -> nn.Sequential:
    layers, c_in = [], 3
    for v in [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]:
        if v == "M":
            layers.append(nn.MaxPool2d(2))
        else:
            layers += [nn.Conv2d(c_in, v, 3, padding=1), nn.ReLU()]
            c_in = v
    c = nn.Sequential(nn.Flatten(), nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(),
                      nn.Linear(4096, 1000))
    return nn.Sequential(nn.Sequential(*layers), c)


def split_conv_dense(model: nn.Module) -> tuple[int, int]:
    """Parameters in convolution layers, and in dense (Linear) layers."""
    conv = sum(n_params(m) for m in model.modules() if isinstance(m, nn.Conv2d))
    dense = sum(n_params(m) for m in model.modules() if isinstance(m, nn.Linear))
    return conv, dense
