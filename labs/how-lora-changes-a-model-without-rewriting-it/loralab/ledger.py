"""Exact parameter and memory arithmetic for the post's ledger (Figures 1 and 4) and the QLoRA bit count.

Nothing here is measured. It is arithmetic on shapes, so the post's numbers can be checked line by line.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1e9


@dataclass(frozen=True)
class Geometry:
    d: int = 4096          # model width
    layers: int = 32
    ffn: int = 11008       # SwiGLU hidden width
    vocab: int = 32000
    kv_dim: int | None = None  # None means full multi-head attention (k and v are d x d)
    tied: bool = False

    def matrices(self) -> dict[str, tuple[int, int]]:
        """(out, in) of the seven adaptable linear maps in one block."""
        kv = self.kv_dim or self.d
        return {"q_proj": (self.d, self.d), "k_proj": (kv, self.d), "v_proj": (kv, self.d), "o_proj": (self.d, self.d),
                "gate_proj": (self.ffn, self.d), "up_proj": (self.ffn, self.d), "down_proj": (self.d, self.ffn)}

    def total_params(self) -> int:
        per_layer = sum(o * i for o, i in self.matrices().values()) + 2 * self.d  # two RMSNorms
        emb = self.vocab * self.d * (1 if self.tied else 2)
        return self.layers * per_layer + emb + self.d


LLAMA2_7B = Geometry()
GPT3_175B = Geometry(d=12288, layers=96, ffn=4 * 12288, vocab=50257)
TARGET_SETS = {
    "Q,V only": ("q_proj", "v_proj"),
    "All attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "All linear": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}


def lora_params(g: Geometry, r: int, targets) -> int:
    """r (d_out + d_in) per adapted matrix, summed over layers."""
    m = g.matrices()
    return g.layers * sum(r * (m[t][0] + m[t][1]) for t in targets)


def break_even_rank(d_out: int, d_in: int) -> float:
    """The rank at which r (d_out + d_in) = d_out d_in. Above it the factors store more numbers than the matrix."""
    return d_out * d_in / (d_out + d_in)


# Bytes per parameter in the post's ledger: bf16 weights and grads, fp32 master copy, fp32 Adam m and v.
FULL_FT_BYTES = {"weights (bf16)": 2, "gradients (bf16)": 2, "master weights (fp32)": 4, "Adam m (fp32)": 4, "Adam v (fp32)": 4}
TRAINABLE_BYTES = sum(FULL_FT_BYTES.values())  # 16
FROZEN_BF16_BYTES = 2


def nf4_bits_per_param(block: int = 64, double_quant: bool = True, dq_block: int = 256,
                       const_bits: int = 32, dq_bits: int = 8) -> float:
    """4-bit codes plus the cost of the per-block absmax constants (QLoRA Sec. 3)."""
    if not double_quant:
        return 4 + const_bits / block
    return 4 + dq_bits / block + const_bits / (block * dq_block)


def training_memory_gb(g: Geometry, method: str, r: int = 8, targets=TARGET_SETS["All linear"]) -> dict[str, float]:
    """Weights + optimizer machinery, no activations. method is full, lora or qlora."""
    n = g.total_params()
    if method == "full":
        return {"frozen": 0.0, "trainable": n * TRAINABLE_BYTES / GB, "total": n * TRAINABLE_BYTES / GB}
    t = lora_params(g, r, targets)
    frozen_bytes = FROZEN_BF16_BYTES if method == "lora" else nf4_bits_per_param() / 8
    out = {"frozen": n * frozen_bytes / GB, "trainable": t * TRAINABLE_BYTES / GB}
    out["total"] = out["frozen"] + out["trainable"]
    return out
