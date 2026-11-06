"""State spaces: why linearity buys parallelism, what time-invariance buys, and what selection costs.

1. Sequential scan = parallel associative scan, in N against ceil(log2 N) dependent rounds.
2. LTI SSM: recurrence = causal convolution with K_k = C a^k B (direct and FFT).
3. Selective SSM: the impulse response depends on where the impulse lands, so there is no single kernel.
4. Mamba-2 view: the same scan is y = M x with a lower-triangular semiseparable matrix (masked attention).
5. a_bar = exp(Delta A) is a forget gate. Linear attention is the same recurrence: checked with C3's code.
"""

import _setup  # noqa: F401

import sys
import time

import torch

from archlab.env import LAB, start
from archlab.scan import causal_conv, lti_kernel, parallel_scan, semiseparable_matrix, sequential_scan

start("06_scans")
g = torch.Generator().manual_seed(0)

print("1. Linear recurrence h_t = a_t h_{t-1} + b_t, batch 1, 256 channels")
print(f"   {'N':>5} {'max |seq - par|':>16} {'seq steps':>10} {'par rounds':>11} {'seq ms':>8} {'par ms':>8}")
for n in [64, 512, 4096]:
    a, b = torch.rand(1, n, 256, generator=g) * 0.5 + 0.5, torch.randn(1, n, 256, generator=g)
    t0 = time.perf_counter()
    hs = sequential_scan(a, b)
    t1 = time.perf_counter()
    hp, rounds = parallel_scan(a, b)
    t2 = time.perf_counter()
    print(f"   {n:>5} {(hs - hp).abs().max().item():>16.1e} {n:>10} {rounds:>11} {1000 * (t1 - t0):>8.1f} "
          f"{1000 * (t2 - t1):>8.1f}")
print("   The parallel scan does about log2 N times more arithmetic. It wins where idle parallel lanes")
print("   exist to absorb that work (a GPU); a few CPU threads may not, so measure before assuming.")
print()

print("2. Time-invariant SSM (diagonal A, 8 channels, state 4): recurrence against convolution")
n, d, s = 200, 8, 4
a_bar = torch.rand(d, s, generator=g) * 0.3 + 0.65
b_bar, c = torch.randn(d, s, generator=g), torch.randn(d, s, generator=g)
x = torch.randn(2, n, d, generator=g)
h = sequential_scan(a_bar.expand(2, n, d, s), b_bar * x[..., None])
y_rec = (h * c).sum(-1)
K = lti_kernel(a_bar, b_bar, c, n)
print(f"   max |recurrence - direct conv| = {(y_rec - causal_conv(x, K)).abs().max().item():.1e}")
print(f"   max |recurrence - FFT conv|    = {(y_rec - causal_conv(x, K, fft=True)).abs().max().item():.1e}")
print()

print("3. Impulse response: put x = 1 at position p, read y at p + 10")
for selective in [False, True]:
    responses = []
    for p in [20, 120]:
        x = torch.zeros(1, n, d)
        x[0, p] = 1.0
        if selective:  # a_t depends on the input at t; here, on a fixed random "token" sequence
            tokens = torch.randn(1, n, d, generator=torch.Generator().manual_seed(5))
            dt = 0.2 * torch.nn.functional.softplus(tokens)[..., None]
            a_t = torch.exp(-dt * torch.arange(1, s + 1))
        else:
            a_t = a_bar.expand(1, n, d, s)
        y = (sequential_scan(a_t, b_bar * x[..., None]) * c).sum(-1)
        responses.append(y[0, p + 10])
    rel = ((responses[0] - responses[1]).norm() / responses[0].norm()).item()
    print(f"   {'selective' if selective else 'LTI':>9}: ||response(p=20) - response(p=120)|| / ||response(p=20)|| = {rel:.2f}")
print("   LTI: one kernel serves every position. Selective: the kernel changes with the content, so no FFT path.")
print()

print("4. Mamba-2 duality: scalar decay per step, state 16")
n, s = 64, 16
a = torch.rand(n, generator=g) * 0.5 + 0.5
B, C, x = torch.randn(n, s, generator=g), torch.randn(n, s, generator=g), torch.randn(n, generator=g)
h = sequential_scan(a[None, :, None].expand(1, n, s), (B * x[:, None])[None])[0]
y_scan = (h * C).sum(-1)
M = semiseparable_matrix(a, B, C)
print(f"   max |scan - M x| = {(y_scan - M @ x).abs().max().item():.1e}; M is lower triangular: "
      f"{bool((M.triu(1) == 0).all())}")
print(f"   M = L * (C B^T), L_ij = prod a: this is causal linear attention with a decay mask. "
      f"rank of the block M[32:, :32] = {torch.linalg.matrix_rank(M[32:, :32]).item()} (<= state {s})")
print()

print("5. a_bar = exp(Delta * A) with A = -1: small Delta holds the state, large Delta overwrites it")
for dt in [0.001, 0.01, 0.1, 1.0, 5.0]:
    print(f"   Delta {dt:>6}: a_bar = {torch.exp(torch.tensor(-dt)).item():.4f}")
sys.path.insert(0, str(LAB.parent.parent / "c03-attention"))  # C3 already builds linear attention as an RNN
from attnzoo.linear import causal_linear_attention_recurrent, elu_feature_map  # noqa: E402

q, k, v = (elu_feature_map(torch.randn(1, 2, 12, 8, generator=g)) for _ in range(3))
y_rnn, (S, _) = causal_linear_attention_recurrent(q, k, v)
mask = torch.ones(12, 12).tril()
y_par = ((q @ k.transpose(-1, -2)) * mask) @ v / ((q @ k.transpose(-1, -2)) * mask).sum(-1, keepdim=True)
print(f"   C3 linear attention, recurrent vs masked parallel: max error {(y_rnn - y_par).abs().max().item():.1e};"
      f" state {tuple(S.shape)} whatever the length")
