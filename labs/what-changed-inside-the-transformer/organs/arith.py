"""The post's arithmetic as plain functions: KV cache bytes, decode intensity, MLA, sliding windows,
the SwiGLU 8/3 rule, MoE combinatorics, logits memory, and the Figure 10 calculator.

Every function returns exact integers where the post quotes exact numbers, so the scripts and tests
can check the post line by line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

KIB, MIB, GIB = 1024, 1024**2, 1024**3


# ---------------------------------------------------------------- KV cache ("The KV cache became the constraint")
def kv_bytes(layers: int, kv_heads: int, head_dim: int, seq: int, batch: int = 1, bytes_per: int = 2) -> int:
    """bytes_KV = 2 * L * n_kv * d_h * s * b * bytes/elt. The 2 is one tensor for K and one for V."""
    return 2 * layers * kv_heads * head_dim * seq * batch * bytes_per


def decode_intensity(q_heads: int, kv_heads: int, head_dim: int = 128, seq: int = 4096, bytes_per: int = 2) -> float:
    """FLOPs per byte of one decode step of attention, counted the long way.

    FLOPs: q.k for every query head and key (2 d_h s per head), then p.v (another 2 d_h s): 4 n_q d_h s.
    Bytes: the K and V of every KV head, read once: 2 n_kv d_h s values of `bytes_per` bytes.
    The post's closed form is n_q / n_kv for bf16. This function lets a test check that d_h and s cancel.
    """
    flops = 4 * q_heads * head_dim * seq
    bytes_read = 2 * kv_heads * head_dim * seq * bytes_per
    return flops / bytes_read


def ridge_point(peak_flops: float, bandwidth_bytes: float) -> float:
    """FLOPs per byte at which a kernel stops being memory-bound."""
    return peak_flops / bandwidth_bytes


def mla_bytes_per_token(layers: int, latent: int, rope_dim: int, bytes_per: int = 2) -> int:
    """MLA caches one latent c_t (r values) and one shared RoPE key (d_r values) per layer per token."""
    return layers * (latent + rope_dim) * bytes_per


def mha_equivalent_bytes_per_token(layers: int, heads: int, qk_dim: int, v_dim: int, bytes_per: int = 2) -> int:
    """What full multi-head attention with the same head shapes would cache: K (qk_dim) and V (v_dim) per head."""
    return layers * heads * (qk_dim + v_dim) * bytes_per


def swa_bytes(global_layers: int, local_layers: int, window: int, seq: int, per_layer_token: int) -> int:
    """Global layers cache all `seq` tokens. Local (sliding-window) layers cache at most `window`."""
    return global_layers * seq * per_layer_token + local_layers * min(seq, window) * per_layer_token


# ---------------------------------------------------------------- FFN ("The 8/3 accounting")
def swiglu_parity_width(d: int, multiple: int = 256) -> tuple[float, int]:
    """Width that gives 3 d d_ff = 8 d^2, before and after rounding up to a multiple."""
    exact = 8 * d / 3
    return exact, multiple * math.ceil(exact / multiple)


# ---------------------------------------------------------------- MoE ("Why many small experts beat few large ones")
def routing_configurations(experts: int, top_k: int) -> int:
    return math.comb(experts, top_k)


# ---------------------------------------------------------------- vocabulary ("The vocabulary quadrupled")
def logits_bytes(batch: int, seq: int, vocab: int, bytes_per: int = 4) -> int:
    """The (B, S, V) logits tensor that cross-entropy needs, in fp32 by default."""
    return batch * seq * vocab * bytes_per


# ---------------------------------------------------------------- Figure 10, ported from the post's figures.js
@dataclass(frozen=True)
class Spec:
    label: str
    d: int
    L: int
    nq: int
    nkv: int
    dh: int
    V: int
    tied: bool
    dff: int
    glu: bool = True
    moe: bool = False
    N: int = 0  # routed experts
    k: int = 0  # experts per token
    de: int = 0  # expert width
    shared: int = 0
    dense_layers: int = 0  # leading layers that keep a dense FFN (DeepSeek-V3 has 3)
    attn: str = "gqa"  # "gqa" covers MHA, GQA and MQA; "mla" uses the latent cache
    s: int = 4096


# DeepSeek-V3's MLA shapes, the same constants the post's calculator uses.
MLA_SHAPE = {"r": 512, "rope": 64, "qlora": 1536, "nope": 128, "vh": 128}

PRESETS = {
    "gpt2xl": Spec("GPT-2 XL", 1600, 48, 25, 25, 64, 50257, True, 6400, glu=False, s=1024),
    "llama2": Spec("Llama 2 7B", 4096, 32, 32, 32, 128, 32000, False, 11008, s=4096),
    "llama3": Spec("Llama 3 8B", 4096, 32, 32, 8, 128, 128256, False, 14336, s=8192),
    "olmo3": Spec("Olmo 3 32B", 5120, 64, 40, 8, 128, 100352, False, 27648, s=65536),
    "qwen3": Spec("Qwen3-235B-A22B", 4096, 94, 64, 4, 128, 151936, False, 12288, moe=True, N=128, k=8, de=1536, s=32768),
    "gptoss": Spec("gpt-oss-120b", 2880, 36, 64, 8, 64, 201088, False, 2880, moe=True, N=128, k=4, de=2880, s=131072),
    "dsv3": Spec("DeepSeek-V3", 7168, 61, 128, 128, 128, 129280, False, 18432, moe=True, N=256, k=8, de=2048,
                 shared=1, dense_layers=3, attn="mla", s=131072),
}


def calc_model(c: Spec) -> dict[str, float]:
    """Parameters, active parameters, KV bytes per token and FLOPs per token, as in the post's Figure 10.

    P_attn = 2 L d d_h (n_q + n_kv); P_ffn = 3 L d d_ff (dense) or 3 L d d_e (N + shared) (MoE);
    FLOPs/token ~ 2 P_active + 4 L n_q d_h s. Norm gains and biases are not counted.
    """
    emb = c.V * c.d * (1 if c.tied else 2)
    if c.attn == "mla":
        m = MLA_SHAPE
        per = (c.d * m["qlora"] + m["qlora"] * c.nq * (m["nope"] + m["rope"]) + c.d * (m["r"] + m["rope"])
               + m["r"] * c.nq * (m["nope"] + m["vh"]) + c.nq * m["vh"] * c.d)
        attn = per * c.L
        kv_per_tok = mla_bytes_per_token(c.L, m["r"], m["rope"])
    else:
        attn = 2 * c.L * c.d * c.dh * (c.nq + c.nkv)
        kv_per_tok = kv_bytes(c.L, c.nkv, c.dh, seq=1)
    if c.moe:
        lm = c.L - c.dense_layers
        ffn_total = 3 * lm * c.d * c.de * (c.N + c.shared) + 3 * c.dense_layers * c.d * c.dff + lm * c.d * c.N
        ffn_active = 3 * lm * c.d * c.de * (c.k + c.shared) + 3 * c.dense_layers * c.d * c.dff
    else:
        ffn_total = ffn_active = (3 if c.glu else 2) * c.L * c.d * c.dff
    total = emb + attn + ffn_total
    active = attn + ffn_active + c.V * c.d  # the LM head runs on every token; the embedding lookup is not a matmul
    attn_flops = 4 * c.L * c.nq * c.dh * c.s
    flops = 2 * active + attn_flops
    return {"total": total, "active": active, "frac": active / total, "kv_per_tok": kv_per_tok,
            "kv_at_s": kv_per_tok * c.s, "flops": flops, "attn_share": attn_flops / flops}


def with_mha(c: Spec) -> Spec:
    """The same model with one KV head per query head."""
    return replace(c, nkv=c.nq)
