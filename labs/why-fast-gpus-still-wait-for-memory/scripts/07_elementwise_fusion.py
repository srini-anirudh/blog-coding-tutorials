"""Elementwise ops are priced by bytes, and fusion removes bytes. GPU, bf16.

Part 1 times single elementwise ops on a (8, 8192, 8192) activation and compares each with t = bytes / beta.
Part 2 runs the post's chain (residual +, RMSNorm, scale, SiLU, gate *) as Triton kernels, with every choice of
kernel boundaries, and compares measured time with the traffic model in rooflab.model.chain_tensor_passes.
"""

import itertools

import _setup  # noqa: F401
import torch
import torch.nn.functional as F

from rooflab import kernels
from rooflab import model as M
from rooflab.report import measured_ceilings, save_json, start
from rooflab.timing import auto_time

start("07_elementwise_fusion")
c = measured_ceilings()
beta = c["beta"]
print(f"beta = {beta / 1e12:.2f} TB/s ({c['source']})")
dev, bf = "cuda", torch.bfloat16

shape = (8, 8192, 8192)
x = torch.randn(shape, device=dev, dtype=bf)
r = torch.randn(shape, device=dev, dtype=bf)
w = torch.randn(shape[-1], device=dev, dtype=bf)
n = x.numel()
print(f"\n## Part 1: single ops on {shape} bf16 ({n * 2 / 1e9:.2f} GB per tensor)")
print(f"{'op':>26} {'bytes/elt':>9} {'model ms':>9} {'measured ms':>12} {'measured / model':>17}")
ops = [
    ("residual add x + r", 6, lambda: x + r),
    ("cast bf16 -> fp32", 6, lambda: x.float()),
    ("SiLU", 4, lambda: F.silu(x)),
    ("GELU (tanh)", 4, lambda: F.gelu(x, approximate="tanh")),
    ("softmax over last dim", 4, lambda: torch.softmax(x, -1)),
    ("RMSNorm F.rms_norm", 4, lambda: F.rms_norm(x, (shape[-1],), w)),
]
part1 = []
for name, bpe, fn in ops:
    t = auto_time(fn, budget=0.1)
    model_t = bpe * n / beta
    part1.append({"op": name, "bytes_per_elt": bpe, "t": t, "model": model_t})
    print(f"{name:>26} {bpe:9d} {model_t * 1e3:9.3f} {t * 1e3:12.3f} {t / model_t:17.2f}")
del x, r

tokens, d = 65536, 8192
print(f"\n## Part 2: the fusion chain, {tokens} tokens x d = {d}, bf16 ({tokens * d * 2 / 1e9:.2f} GB per tensor)")
x = torch.randn(tokens, d, device=dev, dtype=bf)
r = torch.randn(tokens, d, device=dev, dtype=bf)
g = torch.randn(tokens, d, device=dev, dtype=bf)
per_pass = tokens * d * 2
print(f"{'boundaries after':>40} {'kernels':>7} {'passes':>6} {'model ms':>9} {'measured ms':>12} {'speed-up':>9}")
part2 = []
for breaks in itertools.product([True, False], repeat=kernels.NSTAGES - 1):
    breaks = list(breaks)
    passes = M.chain_tensor_passes(breaks)
    t = auto_time(lambda: kernels.chain(x, r, g, w, breaks), budget=0.1)
    part2.append({"breaks": breaks, "kernels": sum(breaks) + 1, "passes": passes, "t": t,
                  "model": passes * per_pass / beta})
base = part2[0]["t"]  # all boundaries on
for row in sorted(part2, key=lambda q: -q["t"]):
    names = ", ".join(kernels.STAGE_NAMES[i] for i, b in enumerate(row["breaks"]) if b) or "none (fully fused)"
    print(f"{names:>40} {row['kernels']:7d} {row['passes']:6d} {row['model'] * 1e3:9.2f} {row['t'] * 1e3:12.2f} "
          f"{base / row['t']:8.2f}x")
fused = next(q for q in part2 if not any(q["breaks"]))
print(f"\nAll split -> fully fused: model {part2[0]['passes'] / fused['passes']:.2f}x, "
      f"measured {base / fused['t']:.2f}x. Achieved bandwidth fully fused: "
      f"{fused['passes'] * per_pass / fused['t'] / 1e12:.2f} TB/s.")


def eager(x, r, g, w):
    h = x + r
    v = F.rms_norm(h, (d,), w)
    return h, F.silu(v) * g


t_eager = auto_time(lambda: eager(x, r, g, w), budget=0.1)
compiled = torch.compile(eager)
t_comp = auto_time(lambda: compiled(x, r, g, w), budget=0.1)
print(f"Same chain in eager PyTorch: {t_eager * 1e3:.2f} ms. torch.compile: {t_comp * 1e3:.2f} ms. "
      f"Triton fully fused: {fused['t'] * 1e3:.2f} ms.")
save_json("07_elementwise_fusion", {"beta": beta, "part1": part1, "part2": part2,
                                    "eager_s": t_eager, "compile_s": t_comp})
