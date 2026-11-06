"""Axis 2, training cost, on CPU: where attention's N^2 d term passes the FFN's N d^2 term, and the
sequential wall of a recurrence.

1. One layer's forward, d = 256: attention core (QK^T, softmax, AV) against the FFN (d -> 4d -> d).
   FLOPs: attention 4 N^2 d, FFN 16 N d^2, equal at N = 4d = 1024.
2. nn.RNN over N steps against one matmul doing the same multiply-adds: same work, N dependent steps.
"""

import _setup  # noqa: F401

import time

import torch
import torch.nn.functional as F

from archlab.env import start

start("08_training_cost", threads=4)


def bench(fn, reps=7):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1000 * sorted(ts)[len(ts) // 2]


d = 256
W1, W2 = torch.randn(d, 4 * d) / d ** 0.5, torch.randn(4 * d, d) / (4 * d) ** 0.5
print(f"1. Forward of one layer, batch 1, d = {d}, fp32 (median of 7)")
print(f"   {'N':>5} {'attention ms':>13} {'FFN ms':>8} {'time ratio':>11} {'FLOP ratio N/(4d)':>18}")
for n in [256, 512, 1024, 2048, 4096]:
    x = torch.randn(n, d)
    q, k, v = torch.randn(3, 1, 4, n, d // 4).unbind(0)
    t_att = bench(lambda: torch.softmax(q @ k.transpose(-1, -2) / 8, -1) @ v)
    t_ffn = bench(lambda: F.gelu(x @ W1) @ W2)
    print(f"   {n:>5} {t_att:>13.2f} {t_ffn:>8.2f} {t_att / t_ffn:>11.2f} {n / (4 * d):>18.2f}")
print("   FLOPs predict equal cost at N = 4d. Past that, the time ratio grows faster than the FLOP ratio:")
print("   the N x N score matrix (4 heads x N^2 floats) is written and read in full, the memory traffic that")
print("   FlashAttention removes (C3 builds it). On a shared CPU the crossover point itself moves between runs.")
print()

print("2. The sequential wall: nn.RNN (tanh, d = 256) over N steps against one batched matmul of equal work")
rnn = torch.nn.RNN(d, d, batch_first=True)
Wh = torch.randn(d, d)
print(f"   {'N':>5} {'RNN ms':>8} {'one matmul ms':>14} {'ratio':>6}")
with torch.no_grad():
    for n in [256, 1024, 4096]:
        x = torch.randn(1, n, d)
        X = torch.randn(n, d)
        t_rnn = bench(lambda: rnn(x))
        t_mm = bench(lambda: torch.tanh(X @ Wh + X @ Wh))  # the two d x d matmuls per step, all at once
        print(f"   {n:>5} {t_rnn:>8.2f} {t_mm:>14.2f} {t_rnn / t_mm:>6.1f}")
print("   Same multiply-adds; the recurrence must wait for h_{t-1} at every step.")
