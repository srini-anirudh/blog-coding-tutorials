# Lab: How Attention Moves Information

A code companion to the post [How Attention Moves Information](https://srini-anirudh.github.io/blogs/how-attention-moves-information/).
The post derives attention from one question: how should a position choose what to read? This lab runs the
post's central claims as small experiments, so you can watch each one happen.

The [C3 attention tutorial](../../c03-attention/) already builds every layer the post names: MHA, GQA, MQA, MLA,
masks, windows, linear attention and tiled online softmax. This lab does not rebuild those layers. It tests the
claims the post makes about them. Where C3 already has the right module, the lab imports it.

## What the lab shows

1. **Why three projections.** Raw scores `x_i . x_j` are symmetric and favour the token itself. A shared map
   `W x` is still symmetric. Separate `W_Q` and `W_K` can learn a directed relation. A three-way cycle task shows
   the difference: 100% against about 45%.
2. **Where the square root comes from.** The logit spread grows as `sqrt(d_k)`, softmax entropy collapses without
   the scale, and dividing by `sqrt(d_k)` holds the logit spread and the entropy flat across head widths.
3. **Masks and reach.** A missing document mask leaks between packed documents. Stacked sliding windows reach
   `L (w - 1)` back, checked by mask powers and by autograd through real layers. The script also counts how many
   layers sparse and grid patterns need before every input can reach every output.
4. **Why more than one head.** At the same width and parameter count, one head learns one lookup. Two heads learn
   two. Pruning shows which heads carry the work.
5. **The cost arithmetic.** KV cache size, decode FLOPs per byte, tensor-parallel replication, MLA cache size and
   ViT token counts, computed from published model shapes.
6. **Linear attention and recall.** A fixed `d x d` state against a growing softmax cache, on associative recall.
7. **MLA and RoPE.** Absorbing `W_UK` into the query is exact. RoPE on the key breaks it. Decoupled RoPE, as in
   C3's module, makes it exact again.
8. **FlashAttention is not a new rule.** A tiled kernel and a textbook kernel both run through a counter on every
   memory access. They return the same numbers and move very different numbers of bytes. A GPU run compares
   PyTorch's math and flash kernels on an H100.

## Setup

From the repository root, after `pip install -e ".[dev]"`. Scripts cap PyTorch at 8 threads. Tests use 2.
Nothing downloads. Script 09 needs a CUDA GPU. Everything else runs on CPU.

## Run everything

```bash
python labs/how-attention-moves-information/scripts/01_projections.py     # about 1 min, trains small models
python labs/how-attention-moves-information/scripts/02_scaling.py
python labs/how-attention-moves-information/scripts/03_masks_and_reach.py
python labs/how-attention-moves-information/scripts/04_heads.py           # about 1 min, trains small models
python labs/how-attention-moves-information/scripts/05_budgets.py
python labs/how-attention-moves-information/scripts/06_linear_recall.py
python labs/how-attention-moves-information/scripts/07_mla_rope.py
python labs/how-attention-moves-information/scripts/08_flash_io.py
python labs/how-attention-moves-information/scripts/09_flash_gpu.py       # CUDA only, about 1 min

python -m pytest labs/how-attention-moves-information -m "not slow and not gpu and not network"   # about 15 s
python -m pytest labs/how-attention-moves-information -m gpu                                      # CUDA only
```

Each script prints its tables and saves them to `results/<script>.txt`. Each file starts with its hardware and
command.

## Results

CPU rows: **AMD EPYC 7R13, 8 threads, torch 2.15.0, CPU only**. GPU rows: **NVIDIA H100 80GB HBM3, torch 2.15.0,
CUDA 13.2**. Training runs use fixed seeds. Other seeds can move the accuracies by a few points.

### 1. Score rules on directional tasks (`01_projections.py`, CPU)

One attention layer, d = 32, 1500 Adam steps, mean test accuracy of 2 seeds.

| Task | Raw `x . x` | Shared `W` | Separate `W_Q`, `W_K` | Cap for symmetric scores |
|---|---|---|---|---|
| Three-way cycle (chance 0.125) | 0.481 | 0.449 | 1.000 | 2 of 3 rows |
| Previous token, 12 positions (chance 0.083) | 0.691 | 0.919 | 1.000 | 11 of 12 rows, about 0.924 |

With no parameters, `softmax(X X^T)` on 64 random vectors in R^64 puts its largest weight on the token itself in
every row. The mean self weight is 1.000 unscaled and 0.923 scaled by `1/sqrt(d)`.

The symmetry argument is narrower than it first sounds. Softmax normalises each row on its own, so symmetric
scores still give directional weights. What they cannot hold is a directed cycle of preferences. The
previous-token task has one cycle of 12, so a shared map loses only one row. The three-way cycle loses one row in
three, and the trained shared-map models land below even that cap.

A rank-2 `W_Q W_K^T` already solves the previous-token task (1.000 at d_k = 2, 0.410 at d_k = 1). Replacing
`(W_Q, W_K)` by `(W_Q A, W_K A^-T)` changes the weights by up to 12 and the scores by 5.7e-13. Shuffling the
inputs shuffles the outputs (2.4e-7) until positions are added (4.17).

### 2. The square root (`02_scaling.py`, CPU)

128 keys, 2000 queries, N(0, 1) entries. The uniform entropy is ln 128 = 4.85 nats.

| d_k | Logit std, unscaled | Entropy, unscaled | Rows with max weight > 0.99, unscaled | Entropy, scaled | Softmax Jacobian norm, unscaled / scaled |
|---|---|---|---|---|---|
| 16 | 4.04 | 1.62 | 0.017 | 4.36 | 0.293 / 0.137 |
| 128 | 11.27 | 0.44 | 0.297 | 4.37 | 0.174 / 0.135 |
| 512 | 22.59 | 0.21 | 0.575 | 4.37 | 0.101 / 0.136 |
| 8192 | 90.76 | 0.05 | 0.862 | 4.37 | 0.031 / 0.136 |

The post's numbers hold: the logit std is `sqrt(d_k)`, which is 11.3 at d_k = 128, and the scale keeps it at 1.
The vanishing-gradient part needs care. At d_k = 128 the unscaled Jacobian (0.174) is still larger than the scaled
one (0.135), because a near-uniform softmax also has a small Jacobian. It falls well below only from about
d_k = 512. The clear effect at moderate widths is that attention sharpness depends on d_k unless you scale.

### 3. Masks and reach (`03_masks_and_reach.py`, CPU)

- Packed documents, one layer: changing document 0 moves later documents by up to 1.85 with only a causal mask,
  and by exactly 0 with causal AND same-document.
- Padding AND causal: real tokens match each sequence run alone to 3.6e-7.
- Stacked causal windows reach exactly `L (w - 1)` back, by mask powers and by autograd, for every w and L tried.
  The post's `L w` counts the query as one of the steps. For Mistral 7B (w = 4096, 32 layers) that is 131,072
  against 131,040 counted exactly. The 131K figure is computed, not run.

Layers until every output can depend on every input, n = 256, bidirectional:

| Pattern | Density | Layers |
|---|---|---|
| Band, w = 16 | 0.117 | 17 |
| Strided: block of 16 plus every 16th column | 0.121 | 2 |
| Band w = 16 plus 2 global tokens | 0.132 | 2 |
| Band w = 16 plus 3 random keys per query | 0.128 | 4 |
| Block-diagonal, blocks of 16 | 0.062 | never |
| 16 x 16 grid, 4 x 4 windows, no shift | 0.062 | never |
| 16 x 16 grid, 4 x 4 windows, shift by 2 every other layer | 0.062 | 4 |
| 16 x 16 grid, axial: rows then columns | 0.062 | 2 |

### 4. Heads (`04_heads.py`, CPU)

d_model = 32, 12 positions, 12 tokens, 2000 Adam steps, seed 0. Chance is 0.083. Each position must output the
tokens at i - 1 and at i - 3.

| Heads | d_h | Parameters | Accuracy, i - 1 | Accuracy, i - 3 | Weight on i - 1 / i - 3, per head |
|---|---|---|---|---|---|
| 1 | 32 | 5,656 | 0.996 | 0.111 | 0.53 / 0.43 |
| 2 | 16 | 5,656 | 1.000 | 1.000 | head 0: 0.99 / 0.00, head 1: 0.00 / 0.99 |
| 4 | 8 | 5,656 | 1.000 | 1.000 | heads 0 and 2 look at i - 1, heads 1 and 3 at i - 3 |

The single head splits its weight between both targets. Its output is then a sum of two values, and a linear sum
does not record which token sat where. It reads the larger share (i - 1) and is near chance on the other.

Pruning the trained 4-head model, with no retraining: keeping one head of each kind keeps both lookups (0.87 to
1.00). Keeping two heads of the same kind leaves the other lookup at chance (0.08 to 0.11). This toy echoes
Michel et al. (2019) on redundant heads. It does not reproduce their results on trained translation or BERT
models.

### 5. Cost arithmetic (`05_budgets.py`, computed, not measured)

| Llama-2-7B shape, fp16, 8192 tokens | Hkv | GiB per sequence | GiB for 64 sequences | Decode FLOPs per cache byte |
|---|---|---|---|---|
| MHA | 32 | 4.000 | 256 | 1 |
| GQA | 8 | 1.000 | 64 | 4 |
| MQA | 1 | 0.125 | 8 | 32 |

The fp16 weights are 12.6 GiB, so 64 MHA caches are 20 times the weights. With Hkv = 8 and 8-way tensor
parallelism, each rank holds one KV head. At 16 ranks each KV head is stored twice.

DeepSeek-V2's attention shape caches 576 numbers per token per layer with MLA, against 32,768 for MHA of the
same shape and 2,048 for GQA with 8 KV heads. That is 98.2% smaller than MHA. The paper's 93.3% compares against
its own dense 67B model, a different baseline. We quote it and did not reproduce it.

ViT with 14-pixel patches: 256 tokens at 224 pixels, and 5,329 tokens at 1024 pixels, which is 28.4 million
score entries per head per layer.

### 6. Linear attention recall (`06_linear_recall.py`, CPU, no training)

d = 64, mean of 3 seeds. Softmax uses beta = 32 on unit keys.

| Pairs n | Softmax | Linear, phi = identity | elu + 1 | Softmax cache floats | Linear state floats |
|---|---|---|---|---|---|
| 64 | 1.000 | 1.000 | 0.016 | 8,192 | 4,160 |
| 256 | 1.000 | 0.743 | 0.004 | 32,768 | 4,160 |
| 1024 | 1.000 | 0.090 | 0.001 | 131,072 | 4,160 |
| 4096 | 1.000 | 0.006 | 0.0002 | 524,288 | 4,160 |

The largest n with linear recall at 0.9 or better is 48 at d = 32, 192 at d = 64 and 688 at d = 128. On these
untrained unit keys, the positive elu + 1 map sits at chance, 1 / n: every `phi(k) . phi(k')` is large, so the read
returns close to the mean value. Scaling the keys by 32 first helps only for small n (0.854 at n = 16). Learned
projections would do better. This shows the limit of a fixed state. It is not a trained linear-attention model.

### 7. MLA and RoPE (`07_mla_rope.py`, CPU, float64)

| Check | Result |
|---|---|
| Absorbed `W_UK` against explicit keys, no RoPE | max difference 1.1e-14 |
| Naive absorption with RoPE on the full key | max error 31.0, while the scores have std 5.7 |
| Best single fixed matrix in place of `R_delta W_UK`, offsets 0 to 127 | mean relative error 0.674 |
| C3's decoupled-RoPE MLA, absorbed against explicit | 3.3e-16 |
| C3's decoupled-RoPE MLA, absorbed and cached against full | 3.3e-16 |

### 8. FlashAttention (`08_flash_io.py` on CPU, `09_flash_gpu.py` on GPU)

Slow-memory traffic per head, d = 64, counted on every load and store of kernels that really compute attention:

| n | Textbook (elements) | Tiled, B = 64 | Tiled, B = 128 | Textbook / tiled B = 128 |
|---|---|---|---|---|
| 256 | 327,680 | 164,096 | 98,560 | 3.3 |
| 1024 | 4,456,448 | 2,229,248 | 1,180,672 | 3.8 |
| 2048 | 17,301,504 | 8,652,800 | 4,458,496 | 3.9 |

Every count equals its formula: `4n^2 + 4nd` for the textbook kernel, and `2nd + 2n^2 d / B + n` for the tiled
one. The ratio tends to `2B / d`. A bigger on-chip block saves more, and hardware SRAM sets how big B can be. Our
tiled output matches the textbook one to 8e-16 in float64. In fp32 the two differ by 1.6e-7 to 2.1e-7, which is
smaller than the textbook kernel's own fp32 error against fp64 (5.3e-7). Backward needs `P` (n^2 numbers)
from the textbook kernel. The tiled kernel keeps O and the log-sum-exp (n d + n), and at n = 8192, d = 128 that is
about 64 times fewer numbers.

PyTorch SDPA on **NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2**, bf16, causal, B = 4, H = 16, d = 128. Median of 10
calls after 3 warm-up calls. The math backend materialises the scores. The flash backend does not. "Peak extra"
is the peak allocation above the inputs. The last column is the largest difference from an fp32 math reference.

| n | Math ms | Flash ms | Math peak extra MiB | Flash peak extra MiB | Max diff, math / flash |
|---|---|---|---|---|---|
| 1024 | 1.96 | 0.14 | 724 | 32 | 0.008 / 0.010 |
| 2048 | 7.08 | 0.31 | 2,608 | 65 | 0.008 / 0.009 |
| 4096 | 28.62 | 0.94 | 9,856 | 129 | 0.008 / 0.008 |
| 8192 | 119.84 | 3.27 | 38,273 | 258 | 0.008 / 0.009 |
| 16384 | out of memory | 12.86 | | 516 | not applicable / 0.009 |

Math memory grows 4 times per doubling of n and flash memory grows 2 times. Both outputs sit about 0.01 from the
fp32 reference, which is bf16 rounding. The GPU job ran alone on its GPU. It ran once, so there is no spread to
report. The out-of-memory row shows the score matrix at work: at n = 16384 the scores alone are 32 GiB in bf16.
The `gpu` test (`test_flash_matches_math_on_cuda`) passes on this GPU.

## Post sections and scripts

| Post section | Script | C3 module it builds on |
|---|---|---|
| The question attention answers (permutation equivariance) | `01_projections.py` part 5 | |
| Why three projections | `01_projections.py` | |
| Scaled dot-product attention | `02_scaling.py` | `attnzoo/core.py` |
| Masks: who can I see? | `03_masks_and_reach.py` part 1 | `attnzoo/masks.py` |
| Why more than one head | `04_heads.py` | `attnzoo/mha.py` |
| MHA, MQA, GQA (KV cache calculator) | `05_budgets.py` | `attnzoo/kv_cache.py`, `benchmarks/bench_kv_memory.py` |
| Local and sliding-window attention | `03_masks_and_reach.py` part 2 | `attnzoo/attention_window.py` |
| Sparse attention | `03_masks_and_reach.py` part 3 | `benchmarks/bench_window.py` (wall-clock) |
| Linear attention | `06_linear_recall.py` | `attnzoo/linear.py` (reordering, recurrence) |
| Multi-head latent attention | `07_mla_rope.py`, `05_budgets.py` | `attnzoo/mla.py` (imported) |
| Attention over images | `03_masks_and_reach.py` part 4, `05_budgets.py` | `attnzoo/vision.py` |
| FlashAttention is not a new attention rule | `08_flash_io.py`, `09_flash_gpu.py` | `attnzoo/attention_tiled.py` (tested against) |
| Self-attention and cross-attention, Q-Former | not in this lab | `attnzoo/cross.py`, `attnzoo/resampler.py` |

## Layout

```
labs/how-attention-moves-information/
├── attnmoves/       # common, projections, scaling, reach, heads, budgets, recall, latent, io
├── scripts/         # 01 to 09, one per claim group
├── tests/           # test_routing.py, test_costs.py
└── results/         # saved outputs, each headed by hardware and command
```

## What was not measured

- Every paper result at scale. Nothing here trains a language model. The head-pruning numbers of Michel et al.,
  the quality cost of MQA and GQA, DeepSeek-V2's 93.3% cache saving and quality claims, StreamingLLM's sinks, and
  Big Bird's universal-approximation result are quoted in the post and not reproduced.
- Wall-clock speed of block-sparse against scattered sparsity. C3's `bench_window.py` measures masks on dense
  tensors. No block-sparse kernel was written here.
- Linear attention with trained projections or a gated state. The recall test uses fixed random keys.
- FlashAttention-2 and FlashAttention-3 against each other, FP8 paths, and paged attention. Script 09 compares
  only PyTorch's math and flash SDPA kernels.
- The cost arithmetic in section 5 is computed from model shapes. C3 checks the cache formula against real tensors.
