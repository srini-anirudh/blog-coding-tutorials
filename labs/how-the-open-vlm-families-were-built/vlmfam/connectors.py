"""The connector patterns of the post, from scratch, with parameter-count formulas.

    MLP              LLaVA, Pixtral, Kimi-VL         patch -> Linear -> GELU -> Linear
    2 x 2 merge      Qwen2-VL and later              LayerNorm, concatenate 4 neighbours, MLP
    pixel shuffle    Idefics3, SmolVLM, InternVL     fold s x s neighbours into channels, then project
    average pool     Gemma 3                         4 x 4 average pool to 256 tokens, RMSNorm, one matrix
    query resampler  Qwen-VL, MiniCPM-V              K learned queries cross-attend to all patches

Pixel shuffle for Idefics3 and SmolVLM is imported from C7 (`tinyvlm.projector.pixel_shuffle`). InternVL folds
the same 2 x 2 block in a different channel order, so it gets its own function here. The tests check each module
against the Hugging Face implementation it reproduces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "c07-vlm"))
from tinyvlm.projector import pixel_shuffle  # noqa: E402  (C7: Idefics3 / SmolVLM order)

__all__ = ["pixel_shuffle", "internvl_pixel_shuffle", "MLPConnector", "PatchMerger", "AvgPoolConnector",
           "QueryResampler", "formula_params"]


def internvl_pixel_shuffle(x: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """InternVL's version. x: n x side x side x d (as InternVL keeps it) -> n x side/2 x side/2 x 4d.

    InternVL treats the first spatial axis as width, so the four neighbours land in the channels in a different
    order than in Idefics3. The token count and the information are the same.
    """
    n, w, h, c = x.shape
    x = x.reshape(n, w, int(h * scale), int(c / scale))
    x = x.permute(0, 2, 1, 3).reshape(n, int(h * scale), int(w * scale), int(c / scale ** 2))
    return x.permute(0, 2, 1, 3).contiguous()


class MLPConnector(nn.Module):
    """LLaVA-1.5 and later: two linear layers with a GELU. Optional LayerNorm on the input (InternVL)."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int | None = None, pre_norm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_in) if pre_norm else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden or d_out)
        self.fc2 = nn.Linear(d_hidden or d_out, d_out)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class PatchMerger(nn.Module):
    """Qwen2-VL: normalise each patch, then concatenate each 2 x 2 group and run an MLP of width 4 d_v.

    Qwen's image processor already orders patches so that each 2 x 2 group is four consecutive rows, so the merge
    is a plain reshape here.
    """

    def __init__(self, d_vision: int, d_out: int, merge: int = 2):
        super().__init__()
        self.width = d_vision * merge * merge
        self.ln = nn.LayerNorm(d_vision, eps=1e-6)
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, d_out)

    def forward(self, x):  # x: (num_patches, d_vision), groups of merge^2 consecutive patches
        return self.fc2(F.gelu(self.fc1(self.ln(x).reshape(-1, self.width))))


class AvgPoolConnector(nn.Module):
    """Gemma 3: a 64 x 64 patch grid average-pooled 4 x 4 to 16 x 16 = 256 tokens, RMSNorm, then one matrix."""

    def __init__(self, d_vision: int, d_out: int, patches_per_side: int = 64, tokens: int = 256, eps: float = 1e-6):
        super().__init__()
        self.side, self.kernel = patches_per_side, patches_per_side // int(tokens ** 0.5)
        self.norm_weight = nn.Parameter(torch.zeros(d_vision))  # Gemma's RMSNorm scales by (1 + w)
        self.proj = nn.Parameter(torch.zeros(d_vision, d_out))  # zero-initialised, as in the HF port
        self.eps = eps

    def forward(self, x):  # x: (n, side*side, d)
        n, _, d = x.shape
        grid = x.transpose(1, 2).reshape(n, d, self.side, self.side)
        pooled = F.avg_pool2d(grid, self.kernel).flatten(2).transpose(1, 2)
        p = pooled.float()
        normed = p * torch.rsqrt(p.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.norm_weight.float())
        return (normed.type_as(x)) @ self.proj


class QueryResampler(nn.Module):
    """Qwen-VL / MiniCPM-V: K learned queries read all patches through one multi-head cross-attention layer.

    The output length is K whatever the image size: this is the "learned-query resampler" of the post.
    """

    def __init__(self, d_vision: int, d_model: int, num_queries: int = 64, heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.kv_proj = nn.Linear(d_vision, d_model, bias=False)
        self.ln_q, self.ln_kv, self.ln_post = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.proj = nn.Parameter(torch.randn(d_model, d_model) * d_model ** -0.5)

    def forward(self, x):  # x: (n, num_patches, d_vision) -> (n, num_queries, d_model)
        kv = self.ln_kv(self.kv_proj(x))
        q = self.ln_q(self.query).unsqueeze(0).expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self.ln_post(out) @ self.proj


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _get(cfg: dict, *path, default=None):
    for p in path:
        if not isinstance(cfg, dict) or p not in cfg:
            return default
        cfg = cfg[p]
    return cfg


FORMULA_TYPES = ("llava", "llava_next", "llava_onevision", "qwen2_vl", "qwen2_5_vl", "internvl", "idefics3", "gemma3",
                 "minicpmv")


def formula_params(model_type: str, cfg: dict) -> tuple[int, str] | None:
    """Connector parameters predicted from a config.json alone, with the formula used.

    Returns None for model types this lab does not model.
    """
    v = _get(cfg, "vision_config", "hidden_size")
    t = _get(cfg, "text_config", "hidden_size") or cfg.get("hidden_size")
    if model_type == "llava":
        return v * t + t + t * t + t, f"Linear({v}->{t}) + Linear({t}->{t})"
    if model_type in ("llava_next", "llava_onevision"):
        return v * t + t + t * t + t + t, f"Linear({v}->{t}) + Linear({t}->{t}) + image_newline({t})"
    if model_type in ("qwen2_vl", "qwen2_5_vl"):
        vc = cfg["vision_config"]
        dv = vc.get("embed_dim") if model_type == "qwen2_vl" else vc["hidden_size"]
        out = vc.get("hidden_size") if model_type == "qwen2_vl" else vc["out_hidden_size"]
        m = vc.get("spatial_merge_size", 2)
        w = dv * m * m
        norm = 2 * dv if model_type == "qwen2_vl" else dv  # Qwen2.5-VL switched to RMSNorm (no bias)
        name = "LN" if model_type == "qwen2_vl" else "RMSNorm"
        return norm + w * w + w + w * out + out, f"{name}({dv}) + Linear({w}->{w}) + Linear({w}->{out})"
    if model_type == "internvl":
        s = int(round(1 / cfg.get("downsample_ratio", 0.5)))
        w = v * s * s
        return 2 * w + w * t + t + t * t + t, f"LN({w}) + Linear({w}->{t}) + Linear({t}->{t})"
    if model_type == "idefics3":
        s = cfg.get("scale_factor", 2)
        return v * s * s * t, f"pixel shuffle x{s * s}, Linear({v * s * s}->{t}, no bias)"
    if model_type == "gemma3":
        return v * t + v, f"RMSNorm({v}) + matrix {v}x{t}"
    if model_type == "minicpmv":
        k, d = cfg["query_num"], cfg["hidden_size"]
        total = k * d + v * d + 3 * 2 * d + (3 * d * d + 3 * d) + (d * d + d) + d * d
        return total, f"{k} queries x {d} + kv Linear({v}->{d}) + attention({d}) + proj {d}x{d} + 3 LN"
    return None
