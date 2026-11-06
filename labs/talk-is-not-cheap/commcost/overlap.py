"""Gradient bucketing during the backward pass, as a timeline model.

Gradients appear in reverse layer order while compute runs. They are packed into buckets. A bucket's AllReduce is
launched once it is full, on one communication stream, so AllReduces run one after another and never before their
bucket is ready. The step ends when both compute and the last AllReduce are done. The exposed tail is the time
communication adds after the backward pass has finished.

This is a model, priced with the ring AllReduce formula. It is not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from .alphabeta import t_ring

PEAK = 989e12  # H100 dense bf16, FLOP/s


@dataclass
class Grad:
    name: str
    params: int


def llama7b_tensors(layers: int = 32, d: int = 4096, ffn: int = 11008, vocab: int = 32000) -> list[Grad]:
    """Parameter tensors of a Llama-style 7B model in forward order (6.74 B parameters)."""
    out = [Grad("embed", vocab * d)]
    for i in range(layers):
        out += [Grad(f"l{i}.attn_norm", d)]
        out += [Grad(f"l{i}.{w}", d * d) for w in ("q", "k", "v", "o")]
        out += [Grad(f"l{i}.mlp_norm", d)]
        out += [Grad(f"l{i}.{w}", d * ffn) for w in ("gate", "up", "down")]
    out += [Grad("final_norm", d), Grad("lm_head", vocab * d)]
    return out


def post_model_tensors(psi: float = 7e9, layers: int = 32) -> list[Grad]:
    """The post's Figure 18 model: 32 equal layers, one gradient blob per layer."""
    return [Grad(f"layer{i}", int(psi / layers)) for i in range(layers)]


def simulate(
    tensors: list[Grad],
    bucket_bytes: float,
    p: int,
    alpha: float,
    beta: float,
    tokens: int = 8192,
    mfu: float = 0.45,
    bytes_per_grad: int = 2,
    overlap: bool = True,
) -> dict:
    """Return compute time, step time, exposed tail and the number of AllReduces launched."""
    rate = PEAK * mfu
    t_compute, filled, ready = 0.0, 0.0, []
    for g in reversed(tensors):
        t_compute += 4 * g.params * tokens / rate  # backward is about 2x forward, forward is 2 * params * tokens
        filled += g.params * bytes_per_grad
        if overlap and filled >= bucket_bytes:
            ready.append((t_compute, filled))
            filled = 0.0
    if filled:
        ready.append((t_compute, filled))
    t = 0.0
    for when, size in ready:
        t = max(t, when) + t_ring(size, p, alpha, beta)
    step = max(t_compute, t)
    return {"compute_s": t_compute, "step_s": step, "exposed_s": step - t_compute, "messages": len(ready)}


def sweep(tensors, bucket_sizes, p, alpha, beta, **kw) -> list[dict]:
    return [{"bucket_bytes": b, **simulate(tensors, b, p, alpha, beta, **kw)} for b in bucket_sizes]


def best_bucket(rows: list[dict]) -> dict:
    """Smallest bucket among those within 0.1% of the lowest exposed time (the curve is flat near its minimum)."""
    lo = min(r["exposed_s"] for r in rows)
    return min((r for r in rows if r["exposed_s"] <= lo * 1.001 + 1e-12), key=lambda r: r["bucket_bytes"])
