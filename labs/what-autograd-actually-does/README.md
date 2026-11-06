# Lab: What Autograd Actually Does

Runnable code for the theory post [What Autograd Actually Does](https://srini-anirudh.github.io/blogs/what-autograd-actually-does/).
The post builds reverse-mode differentiation from the chain rule and follows it into memory. This lab builds each
step small and runs it, so you can watch it happen.

The package `autogradlab` has:

- a scalar reverse-mode engine and a forward-mode dual number, run on the post's one-unit residual network;
- the matrix-chain cost of forward and reverse mode, counted and timed;
- hand-written VJPs for Linear, ReLU, GELU, LayerNorm, RMSNorm, softmax, cross-entropy, residual and attention,
  checked against `torch.autograd`;
- a ledger that measures every tensor autograd saves for the backward pass;
- gradient checkpointing written from scratch;
- custom `torch.autograd.Function` rules, a buggy one, and the two tests that catch it.

Everything runs on CPU in about a minute in total. Nothing is downloaded.

C1 ([`c01-pytorch`](../../c01-pytorch/)) already walks PyTorch's `grad_fn` graph (`scratchnet/autograd_tools.py`,
`scripts/03_autograd.py`) and covers `zero_grad` and the training step. This lab does not repeat that. It covers
what the post adds: the cost argument, the layer VJPs, saved tensors and recomputation.

## Layout

```
labs/what-autograd-actually-does/
├── autogradlab/
│   ├── scalar.py      # Value (reverse mode), Dual (forward mode), the post's example graph
│   ├── chain.py       # matrix-chain cost, both multiplication orders, gradients three ways, dot-product test
│   ├── vjp.py         # forward/backward pairs for every layer, recomputing attention, a matmul counter
│   ├── saved.py       # SavedTensorLedger: bytes saved for backward, live and peak
│   ├── checkpoint.py  # segment checkpointing from scratch
│   ├── custom.py      # MyLinear, BuggyLinear, RoundSTE, dot_test
│   └── report.py      # writes results/<script>.txt with a hardware line and the command
├── scripts/           # 01 to 09, listed below
├── results/           # output of every script, and the test run
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Only `torch` is needed. The numbers below used
`OMP_NUM_THREADS=8`.

## Run everything

```bash
python -m pytest labs/what-autograd-actually-does -m "not slow and not gpu and not network"

python labs/what-autograd-actually-does/scripts/01_scalar_graph.py           # < 1 s
python labs/what-autograd-actually-does/scripts/02_forward_vs_reverse.py     # about 10 s
python labs/what-autograd-actually-does/scripts/03_backward_cost.py          # about 15 s
python labs/what-autograd-actually-does/scripts/04_layer_vjps.py             # < 5 s
python labs/what-autograd-actually-does/scripts/05_softmax_cross_entropy.py  # < 5 s
python labs/what-autograd-actually-does/scripts/06_what_autograd_keeps.py    # < 5 s
python labs/what-autograd-actually-does/scripts/07_custom_backward.py        # < 5 s
python labs/what-autograd-actually-does/scripts/08_checkpointing.py          # about 10 s
python labs/what-autograd-actually-does/scripts/09_precision_and_order.py    # about 5 s
```

Each script prints its results and writes them to `results/<script name>.txt`.

## Which script covers which section of the post

| Post section | Script |
|---|---|
| One number, a hundred billion gradients | `02_forward_vs_reverse.py` (finite differences) |
| The chain rule, taken literally | `01_scalar_graph.py` |
| Everything is a graph; Local derivatives and the adjoint | `01_scalar_graph.py` |
| Forward accumulation, reverse accumulation | `01_scalar_graph.py`, `02_forward_vs_reverse.py` |
| When the wires become buses; JVP and VJP | `02_forward_vs_reverse.py`, `04_layer_vjps.py` (dot-product test) |
| Why reverse mode wins, precisely | `02_forward_vs_reverse.py`, `03_backward_cost.py` |
| The backward pass of every layer you use | `04_layer_vjps.py`, `03_backward_cost.py` (matmul counts) |
| Softmax, and the cancellation that saves it | `05_softmax_cross_entropy.py` |
| What autograd actually keeps | `06_what_autograd_keeps.py` (parts 1 and 2) |
| The API is the graph, exposed | `06_what_autograd_keeps.py` (parts 3 to 6) |
| Writing your own backward | `07_custom_backward.py` |
| Trading memory for recomputation | `08_checkpointing.py`, `06_what_autograd_keeps.py` (attention) |
| Where this lands in a real training step | `09_precision_and_order.py` |

## Results

All numbers are from one machine: AMD EPYC 7R13, 8 threads, torch 2.15.0, CPU only. The machine was shared with
other jobs (load average near 60 on 192 cores). Treat wall times as rough. Counts, bytes and errors do not depend
on load. Raw output is in `results/`.

Tests: 24 passed, none deselected (`results/tests.txt`). There are no `slow`, `gpu` or `network` tests.

| Measurement | Result | File |
|---|---|---|
| Post's graph at w = 1.5, x = 2, b = -1, y = 2: dL/dx, and the part along the skip | 2.1324, of which 1.9281 (90%) | `01` |
| Scalar engine against `torch.autograd`, float64 | identical (0.0 difference) | `01` |
| Fan-out node visited before all children report: d out/da | 1.5729 instead of 2.8213, no error | `01` |
| Jacobian chain 4096 -> 4096 -> 4096 -> 1, multiply-adds: reverse / forward | 3.36e7 / 6.87e10 (2048x) | `02` |
| Same chain at 4096 outputs: forward / reverse | 1.0 | `02` |
| Real chain 2048 -> 2048 -> 2048 -> 1, float64: reverse / forward order | 5.8 / 111.5 ms | `02` |
| Input gradient of a tanh MLP, 1024 inputs: reverse / forward (vmapped) / finite differences | 2.6 / 16.4 / 360 ms | `02` |
| Same, 16 inputs | 1.7 / 3.9 / 4.6 ms | `02` |
| Finite-difference relative error, best step size: float64 / float32 / bfloat16 | 7e-11 / 6e-5 / 0.23 | `02` |
| Matmuls forward / backward: linear layer; first layer (input needs no grad) | 1 / 2; 1 / 1 | `03` |
| Matmuls forward / backward: attention with P saved; P recomputed | 2 / 4; 2 / 5 | `03` |
| FLOPs per token per parameter, 8 linear layers of 2048: forward / backward | 2.00 / 4.00 (3.75 when the input needs no grad) | `03` |
| Wall time, same stack, batch 512: forward / backward | 55 / 107 ms (ratio 1.94) | `03` |
| Hand-written VJPs for 9 layers against `torch.autograd`, float64: largest error | 8.9e-16 | `04` |
| LayerNorm dx: largest dot with 1 / with x_hat (eps = 1e-5) | 9e-16 / 2e-4 | `04` |
| Same at eps = 0; RMSNorm dx dot with 1 | 9e-16 / 2e-15; 6.9 (not orthogonal) | `04` |
| GELU derivative: zero crossing; minimum | x = -0.752; -0.129 at x = -1.414 | `04` |
| Confidently wrong logits, unfused softmax then log: first margin giving nan | float16: 20. float32 and bfloat16: 90 | `05` |
| Fused cross-entropy, same cases | -1 every time (= p - y) | `05` |
| Saved for backward, in activations: matmul / add / mul / ReLU / LayerNorm | 1 / 0 / 2 / 1 / 1.01 | `06` |
| Attention, 4 heads, N = 1024: saved with P kept / P recomputed | 19.0 / 3.0 MiB | `06` |
| Six blocks: saved bytes with first 3 frozen / last 3 frozen | -46% / -23% | `06` |
| Gradient accumulation, 4 microbatches: LayerNorm with 1/K / without 1/K / BatchNorm with 1/K | 1.7e-7 / 3.0 / 0.30 relative error | `06` |
| gradcheck and dot-product gap: correct linear / dW transposed | pass, 1e-9 / FAIL, 9.6 | `07` |
| Learning through round(): true gradient / straight-through | w stays at 0.500 / w reaches 2.700 | `07` |
| Checkpointing 36 layers: peak saved memory, plain backprop | 13.58 MiB, 36 layer evaluations | `08` |
| Same, k = 6 / k = 12 (best) / k = 36 segments | 2.63 / 1.88 / 2.63 MiB, 72 evaluations each | `08` |
| Checkpointed gradients against plain backprop | bitwise identical, every k | `08` |
| 4096 contributions summed in 20 orders, float32: distinct results | 18 (relative spread 3.7e-6) | `09` |
| Loss scaling by 2^16 / by 1000, float32: gradient unchanged bit for bit | yes / no (3e-8) | `09` |
| Gradients near 1e-8 in float16: zero without scaling / with 2^12 / with 2^16 | 100% / 0% / inf (2^16 overflows float16) | `09` |

### What the numbers show that the post does not say

- **The real chain is only 19 times faster in reverse order, not 2048.** The reverse products are matrix-vector
  products. They are limited by reading the matrices from memory, not by arithmetic. The FLOP ratio is exact; the
  time ratio depends on the hardware.
- **Plain autograd runs four matmuls in attention's backward pass, not five.** It keeps P, so it never recomputes
  QK^T. The fifth matmul appears when P is rebuilt in the backward pass, as FlashAttention does.
- **The best checkpoint count was k = 12 for 36 layers, not sqrt(36) = 6.** Each residual MLP layer saves 6 times
  as much as one boundary. Memory then goes like k + 6n/k, which is lowest near k = sqrt(6n) = 14.7.
- **LayerNorm's gradient is orthogonal to x_hat only at eps = 0.** With eps = 1e-5 the dot product is about 2e-4.
  The layer is not quite scale-invariant. Its orthogonality to 1 holds at any eps.
- **float32 breaks before p reaches zero.** At margin 90, p_true = 8e-40 is a subnormal number. The loss is still
  right, but -1/p overflows, and the softmax VJP turns into inf - inf.
- **Freezing late layers saves memory too, just less.** A frozen linear layer no longer needs its input (that is
  only for dW), but it still needs W to pass the gradient back.

### A possible error in the post

The opening section says finite differences for a 100B-parameter model would take "about 1.3 million years" on an
H100 at 1000 TFLOP/s. With the post's own inputs (4e22 FLOPs per token) the arithmetic gives 4e7 seconds, about
1.3 years per token. `02_forward_vs_reverse.py` prints the calculation. The conclusion does not change.

## Not measured

- No GPU runs. The post's ideas are about operation counts and memory, and every count here is exact on CPU. GPU
  timings of forward against backward, and of FlashAttention, were not measured.
- The post's large-scale figures were not reproduced: the Korthikanti et al. activation formula
  s b h (34 + 5 a s / h), their 5x memory reduction and 36% to 4% recompute overhead, the roughly 30% overhead in
  Chen et al., and the 8.6 GB attention matrix at N = 8192. The lab shows the mechanisms at small scale only.
- The Baur-Strassen bound is not tested. `03` measures the backward-to-forward FLOP ratio for linear layers only.
- Peak memory is the bytes autograd saves for the backward pass, counted by hooks. It is not allocator memory, and
  it leaves out parameters, gradients and temporary buffers.
- Multi-threaded scheduling inside PyTorch's autograd engine is not shown. `09` shows only that the order of
  float32 addition changes the result.
- Selective recomputation for a full transformer layer and recursive O(log n) checkpointing are not built.
