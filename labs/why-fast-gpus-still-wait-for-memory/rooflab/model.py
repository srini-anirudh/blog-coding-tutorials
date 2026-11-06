"""The roofline model and the W / Q counts the post derives, as plain functions.

Conventions follow the post. W is FLOPs, with a fused multiply-add counted as 2. Q is bytes across the HBM
boundary, compulsory unless the function name says otherwise. s is bytes per element.
"""

from dataclasses import dataclass


# The model itself.

def attainable(intensity: float, pi: float, beta: float) -> float:
    """P = min(pi, beta * I): the overlapped roofline."""
    return min(pi, beta * intensity)


def attainable_serial(intensity: float, pi: float, beta: float) -> float:
    """No overlap: T = W/pi + Q/beta, so P = 1 / (1/pi + 1/(beta I)). Exactly half the roof at the ridge."""
    return 1.0 / (1.0 / pi + 1.0 / (beta * intensity))


def roofline_time(w: float, q: float, pi: float, beta: float) -> float:
    """T_min = max(W/pi, Q/beta)."""
    return max(w / pi, q / beta)


def fraction_of_peak(intensity: float, ridge: float) -> float:
    """P/pi = min(1, I/I*)."""
    return min(1.0, intensity / ridge)


def little_bytes_in_flight(beta: float, latency_s: float) -> float:
    """Little's law: bytes that must be outstanding to sustain beta."""
    return beta * latency_s


# GEMM.

def gemm_flops(m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k


def gemm_bytes(m: int, n: int, k: int, s: int) -> float:
    """Compulsory: read A and B once, write C once."""
    return float(s * (m * k + k * n + m * n))


def gemm_intensity(m: int, n: int, k: int, s: int) -> float:
    return gemm_flops(m, n, k) / gemm_bytes(m, n, k, s)


def gemm_bytes_tiled(m: int, n: int, k: int, s: int, tm: int, tn: int) -> float:
    """Every Tm x Tn output tile streams its A row-panel and B column-panel from HBM, no cache."""
    tiles = (m // tm) * (n // tn)
    return float(s * tiles * (tm * k + k * tn + tm * tn))


def gemm_intensity_tiled_limit(s: int, tm: int, tn: int) -> float:
    """Large-K limit: I -> 2 / (s (1/Tm + 1/Tn)) = T/s for square tiles."""
    return 2.0 / (s * (1.0 / tm + 1.0 / tn))


def projection_intensity(tokens: int, d: int, s: int) -> float:
    """A d x d weight applied to `tokens` rows: I = 2 M d / (s (2M + d)) (Table 2)."""
    return gemm_intensity(tokens, d, d, s)


# Elementwise and fusion.

def elementwise_intensity(flops_per_elt: float, tensors_rw: int, s: int) -> float:
    """I = f / (s (r + w))."""
    return flops_per_elt / (s * tensors_rw)


def elementwise_time(n: int, tensors_rw: int, s: int, beta: float) -> float:
    """t = s (r + w) n / beta."""
    return s * tensors_rw * n / beta


# Attention, one head, sequence N, head dim d.

def attention_flops(n: int, d: int) -> float:
    return 4.0 * n * n * d


def attention_bytes_naive(n: int, d: int, s: int) -> float:
    """S and P materialised: four N x N round trips plus Q, K, V read and O written."""
    return float(4 * s * (n * d + n * n))


def attention_bytes_compulsory(n: int, d: int, s: int) -> float:
    return float(4 * s * n * d)


def attention_bytes_flash_tiled(n: int, d: int, s: int, br: int) -> float:
    """FA2 loop order: Q read once, K and V re-read once per Q block, O written once."""
    return float(s * (2 * n * d + (n // br) * 2 * n * d))


def decode_attention_intensity(s: int, group: int = 1) -> float:
    """One query vector per head against an N-long cache: I = 2g/s, independent of N."""
    return 2.0 * group / s


# Decode economics (the post's Llama-3-70B example).

def decode_step_time(params: float, s: int, beta: float) -> float:
    """Batch-1 decode reads every weight once per token: t = s P / beta."""
    return s * params / beta


def batch_to_ridge(ridge: float, s: int) -> float:
    """B* = s I* / 2."""
    return s * ridge / 2.0


def kv_bytes_per_token(s: int, d_head: int, h_kv: int, layers: int) -> float:
    return float(2 * s * d_head * h_kv * layers)


def kv_crossover_batch(params: float, s: int, kv_bytes_per_seq: float) -> float:
    """Batch beyond which KV-cache traffic exceeds weight traffic."""
    return s * params / kv_bytes_per_seq


# Optimizers: bytes per parameter, fp32 master weights.

@dataclass(frozen=True)
class OptBytes:
    name: str
    read: int
    write: int
    flops: int

    @property
    def total(self) -> int:
        return self.read + self.write


# The post's accounting includes writing a bf16 copy of the weights for the next forward pass.
ADAM_POST = OptBytes("Adam (post, with bf16 copy)", read=16, write=14, flops=12)


def optimizer_bytes(name: str, bf16_copy: bool = False) -> OptBytes:
    """fp32 state. read/write in bytes per parameter. Optional bf16 weight copy adds 2 written bytes."""
    table = {
        "sgd": (8, 4, 2),  # read p, g; write p
        "sgd_momentum": (12, 8, 4),  # read p, g, buf; write p, buf
        "adam": (16, 12, 12),  # read p, g, m, v; write p, m, v
    }
    r, w, f = table[name]
    return OptBytes(name, r, w + (2 if bf16_copy else 0), f)


# The fusion chain of `kernels.chain`: residual +, RMSNorm, scale, SiLU, gate *.

def chain_tensor_passes(breaks: list[bool]) -> int:
    """Full-size tensors read or written across HBM for a given set of kernel boundaries.

    The group holding the residual add reads x and r and writes h. Every other group reads one intermediate.
    The group holding the gate also reads g. Every group except a lone residual add writes one output.
    The length-d weight vector is ignored.
    """
    groups, start = [], 0
    for i, b in enumerate(breaks):
        if b:
            groups.append((start, i + 1))
            start = i + 1
    groups.append((start, len(breaks) + 1))
    passes = 0
    for s, e in groups:
        passes += 3 if s == 0 else 1  # x, r in and h out, or one intermediate in
        passes += 1 if e == len(breaks) + 1 else 0  # gate read
        passes += 1 if e > 1 else 0  # output written
    return passes
