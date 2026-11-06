"""Axis 3, inference state: a fixed-size state against a cache that keeps every token.

1. Recall capacity. Write n (key, value) pairs into memory, then query each key.
   - KV cache + softmax attention: stores all 2 n d numbers.
   - Linear-attention state S = sum_i v_i k_i^T: d x d numbers, whatever n is.
   The fixed state is exact while the keys are orthogonal (n <= d), then degrades. That is the post's
   "fixed state must compress the past", and the Zoology trade between state size and recall.
2. Decode cost per token on CPU: one attention layer reading an N-token cache against one SSM step.
"""

import _setup  # noqa: F401

import math
import time

import torch

from archlab.env import start

start("05_state_vs_cache")


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def recall(n, d, orthogonal=False, beta=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.linalg.qr(torch.randn(d, d, generator=g))[0][:n] if orthogonal else unit(torch.randn(n, d, generator=g))
    vals = unit(torch.randn(n, d, generator=g))
    S = vals.T @ keys  # (d, d): the whole memory of a linear-attention / SSM layer
    lin = (keys @ S.T)  # read-out for every query at once: (n, d)
    beta = beta or math.sqrt(d) * 2
    att = torch.softmax(beta * keys @ keys.T, -1) @ vals
    acc = lambda out: (unit(out) @ vals.T).argmax(-1).eq(torch.arange(n)).float().mean().item()  # noqa: E731
    return acc(lin), acc(att)


d = 64
print(f"1. Recall of n stored pairs, key and value dimension d = {d} (accuracy = nearest stored value is right)")
print(f"   {'n':>5} {'fixed state, random keys':>25} {'fixed state, orthogonal keys':>29} {'KV cache + softmax':>19}"
      f" {'state floats':>13} {'cache floats':>13}")
for n in [8, 16, 32, 64, 128, 256, 512, 1024]:
    lin, att = recall(n, d)
    orth = f"{recall(n, d, orthogonal=True)[0]:.3f}" if n <= d else "n > d: impossible"
    print(f"   {n:>5} {lin:>25.3f} {orth:>29} {att:>19.3f} {d * d:>13,} {2 * n * d:>13,}")
print()
print("   Same task at n = 256 pairs, growing the fixed state (keys in dimension d_state):")
for ds in [32, 64, 128, 256, 512]:
    lin, _ = recall(256, ds)
    print(f"   d_state {ds:>4}: state {ds * ds:>7,} floats   recall {lin:.3f}")
print()

print("2. Decode one token on CPU, 4 threads, batch 1, fp32, width 1024 (attention: 8 heads of 128)")
torch.set_num_threads(4)


def bench(fn, reps=30):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1000 * sorted(ts)[len(ts) // 2]


H, dh, e, s = 8, 128, 2048, 16
q = torch.randn(1, H, 1, dh)
h_state, a, bx, c = torch.randn(e, s), torch.rand(e, s), torch.randn(e, s), torch.randn(s)


def ssm_step():
    global h_state
    h_state = a * h_state + bx
    return h_state @ c


ssm_ms = bench(ssm_step)
print(f"   {'context':>8} {'attention ms':>13} {'cache MB (K+V)':>15} {'SSM step ms':>12} {'SSM state MB':>13}")
for n in [1024, 4096, 16384, 65536]:
    K, V = torch.randn(1, H, n, dh), torch.randn(1, H, n, dh)
    ms = bench(lambda: torch.softmax(q @ K.transpose(-1, -2) / math.sqrt(dh), -1) @ V)
    print(f"   {n:>8} {ms:>13.3f} {2 * n * H * dh * 4 / 1e6:>15.1f} {ssm_ms:>12.3f} {e * s * 4 / 1e6:>13.3f}")
print("   Medians of 30 runs on a shared machine: read the trend, not the third digit.")
