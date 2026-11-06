"""The cost model from "Where the cost goes": token counts, FLOPs, KV-cache bytes and decode time.

Every function is a formula from the post, written out so that a test can check the post's worked example.
Nothing here is measured. `scripts/03_stage_timing.py` measures the same quantities on a real model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------- token counts

def num_patches(height: int, width: int, patch: int = 14) -> int:
    """N_patch = ceil(H / p) * ceil(W / p) for one image or one tile."""
    return math.ceil(height / patch) * math.ceil(width / patch)


def visual_tokens(height: int, width: int, patch: int = 14, merge: int = 2) -> int:
    """V = N_patch / m^2: tokens the language model sees after an m x m connector merge."""
    return num_patches(height, width, patch) // (merge * merge)


def anthropic_tokens(height: int, width: int) -> int:
    """The rule the post quotes: one visual token per 28 x 28 pixel patch."""
    return num_patches(height, width, 28)


def openai_tokens(height: int, width: int) -> int:
    """GPT-4o "high detail" tiles as the post describes them: fit in 2048 x 2048, shortest side to 768,
    then 170 tokens per 512-pixel tile plus 85 base tokens."""
    scale = min(1.0, 2048 / max(height, width))
    h, w = height * scale, width * scale
    scale = min(1.0, 768 / min(h, w))
    h, w = h * scale, w * scale
    return 170 * math.ceil(h / 512) * math.ceil(w / 512) + 85


def gemini_tokens(height: int, width: int) -> int:
    """258 tokens for a small image (both sides at most 384), else 258 per 768-pixel tile."""
    if height <= 384 and width <= 384:
        return 258
    return 258 * math.ceil(height / 768) * math.ceil(width / 768)


# ---------------------------------------------------------------- model descriptions

@dataclass(frozen=True)
class Encoder:
    params: float          # parameters
    layers: int
    width: int
    global_layers: int     # layers with full attention; the rest are windowed and ignored here
    patch: int = 14
    merge: int = 2

    def flops(self, n_patches: int) -> float:
        """FLOPs_enc ~ 2 * P * N + 4 * L_global * N^2 * d."""
        return 2 * self.params * n_patches + 4 * self.global_layers * n_patches ** 2 * self.width


@dataclass(frozen=True)
class LanguageModel:
    params: float
    layers: int
    width: int
    kv_heads: int
    head_dim: int
    bytes_per_number: int = 2   # BF16

    def prefill_flops(self, tokens: int) -> float:
        """FLOPs_prefill ~ 2 * P * (V + T) + 4 * L * (V + T)^2 * d."""
        return 2 * self.params * tokens + 4 * self.layers * tokens ** 2 * self.width

    def kv_bytes_per_token(self) -> int:
        """2 (key and value) * L * n_kv * d_h * b."""
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_number

    def weight_bytes(self) -> float:
        return self.params * self.bytes_per_number


@dataclass(frozen=True)
class Accelerator:
    name: str
    peak_flops: float        # dense BF16 FLOP/s
    mem_bandwidth: float     # bytes/s
    memory: float            # bytes
    efficiency: float = 0.5  # share of peak reached by compute-bound stages


H100 = Accelerator("H100 SXM (datasheet)", 989e12, 3.35e12, 80e9)

# The post's representative configuration: a 600M encoder with 4 global layers, a 7B GQA language model.
POST_ENCODER = Encoder(params=600e6, layers=32, width=1280, global_layers=4)
POST_LLM = LanguageModel(params=7e9, layers=28, width=3584, kv_heads=4, head_dim=128)


@dataclass
class RequestCost:
    patches: int
    visual: int
    prompt: int
    encoder_flops: float
    prefill_flops: float
    encoder_s: float
    prefill_s: float
    decode_step_s: float
    kv_bytes: float
    image_kv_share: float

    def total_s(self, answer_tokens: int) -> float:
        """Encoder + prefill + answer_tokens decode steps (preprocessing is not modelled)."""
        return self.encoder_s + self.prefill_s + answer_tokens * self.decode_step_s


def request_cost(height: int, width: int, text_tokens: int = 40, enc: Encoder = POST_ENCODER,
                 llm: LanguageModel = POST_LLM, hw: Accelerator = H100, keep: float = 1.0,
                 images: int = 1) -> RequestCost:
    """Cost of one request with `images` copies of an H x W image and a question of `text_tokens` tokens.

    `keep` is the share of visual tokens kept by pruning before the language model.
    """
    patches = num_patches(height, width, enc.patch)
    visual = round(patches // enc.merge ** 2 * keep) * images
    prompt = visual + text_tokens
    e_flops = enc.flops(patches) * images
    p_flops = llm.prefill_flops(prompt)
    achieved = hw.peak_flops * hw.efficiency
    kv = llm.kv_bytes_per_token() * prompt
    return RequestCost(patches=patches * images, visual=visual, prompt=prompt, encoder_flops=e_flops,
                       prefill_flops=p_flops, encoder_s=e_flops / achieved, prefill_s=p_flops / achieved,
                       decode_step_s=(llm.weight_bytes() + kv) / hw.mem_bandwidth, kv_bytes=kv,
                       image_kv_share=visual / prompt)


def concurrent_requests(cost: RequestCost, llm: LanguageModel = POST_LLM, hw: Accelerator = H100,
                        answer_tokens: int = 200, workspace: float = 4e9) -> int:
    """How many such requests fit: (memory - weights - workspace) / KV per request (prompt plus answer)."""
    per_request = llm.kv_bytes_per_token() * (cost.prompt + answer_tokens)
    return max(0, int((hw.memory - llm.weight_bytes() - workspace) // per_request))
