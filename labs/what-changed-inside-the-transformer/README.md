# Lab: What Changed Inside the Transformer?

Code companion for the theory post
[What Changed Inside the Transformer?](https://srini-anirudh.github.io/blogs/what-changed-inside-the-transformer/).

The post takes the 2017 decoder block apart, one organ at a time. For each part it says what replaced it and why.
Most of the reasons are about scale, not quality: memory traffic while decoding, numerical range while training,
and the cost of running every parameter on every token. This lab builds the old and the new version of each part
in small PyTorch, runs them, and recomputes the post's numbers.

The basic modules come from earlier tutorials, so they are not rebuilt here. RMSNorm, SwiGLU, RoPE, the KV cache
and the Switch load-balancing loss come from [C4](../../c04-llm) (`tinyllm`). Masks, linear attention and the full
MLA layer come from [C3](../../c03-attention) (`attnzoo`). This lab adds what the post argues about them.

## Layout

```text
labs/what-changed-inside-the-transformer/
├── organs/
│   ├── arith.py       every formula in the post, and the Figure 10 calculator ported from the post's JavaScript
│   ├── block.py       one decoder with swappable parts: norm, norm placement, position, FFN, QK-norm
│   ├── train.py       a byte-level LM training loop on Tiny Shakespeare
│   ├── position.py    RoPE's offset property, RoPE wavelengths, NoPE attention
│   ├── norms.py       LayerNorm and RMSNorm as reductions, QK-norm logits
│   ├── cache.py       a rolling sliding-window cache, MLA absorption, receptive field through depth
│   ├── moe.py         top-k MoE with no balancing, an auxiliary loss, or DeepSeek-V3's bias rule
│   ├── memory.py      a fixed state as associative memory: additive vs delta-rule writes
│   ├── small.py       chunked cross-entropy, softmax with a sink logit
│   └── report.py      prints a result and saves it with a hardware line
├── scripts/           01 to 11, one per post section (table below)
├── benchmarks/gpu_bench.py   decode bandwidth, norm kernels and cross-entropy memory on a GPU
├── tests/
└── results/           the real output of every script
```

## Setup

From the repository root:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Tested with Python 3.12, PyTorch 2.15 and transformers 5.22. Scripts use the thread count PyTorch picks; set
`OMP_NUM_THREADS` to change it. The CPU numbers below used 8 threads. Tiny Shakespeare (1.1 MB) is read from
`c04-llm/data/` if it is there, otherwise downloaded once into this lab's `data/` folder.

## Run everything

```bash
L=labs/what-changed-inside-the-transformer
python -m pytest $L -m "not slow and not gpu and not network"   # offline, about 15 s
python -m pytest $L -m network                                  # downloads two config.json files
python -m pytest $L -m gpu                                      # needs CUDA, skipped otherwise

python $L/scripts/01_residual_stream.py        # 20 s
python $L/scripts/02_warmup.py                 # 10 runs of 300 steps; 44 min measured on a shared 8-thread CPU
python $L/scripts/02_warmup.py --layers 24 --lr 1e-2 --steps 200 --seeds 1 --runs post:0,post:100,pre:0 --out 02_warmup_deep
python $L/scripts/03_position.py               # seconds
python $L/scripts/04_norms.py                  # 10 s
python $L/scripts/05_ffn_gate.py               # 45 s
python $L/scripts/06_kv_cache.py               # seconds
python $L/scripts/07_sliding_window.py         # seconds
python $L/scripts/08_moe.py                    # 90 s
python $L/scripts/09_beyond_softmax.py         # seconds
python $L/scripts/10_small_changes.py          # 10 s
python $L/scripts/11_spec_calculator.py        # seconds
python $L/scripts/11_spec_calculator.py --hub  # downloads 7 config.json files, no weights
python $L/benchmarks/gpu_bench.py              # needs CUDA, about a minute
```

`02_warmup.py --seeds 1 --steps 150` is a shorter version with less certainty.

## Which script covers which section

| Post section | Script | What it shows |
|---|---|---|
| The residual stream is the object | `01_residual_stream.py` | Stream norm through 24 layers under 4 norm placements; gradient to the first and last block as depth grows |
| Fewer statistics, more placements (axis two) | `02_warmup.py` | 12-layer LM trained with Post-LN, with and without warmup, against Pre-LN, sandwich and post-branch norms. Does not reproduce the failure without warmup (see below) |
| Where the position went | `03_position.py` | RoPE logits depend only on m - n; wavelength of each pair per base; NoPE under a causal mask |
| Fewer statistics (axis one), QK-norm | `04_norms.py` | LayerNorm = centering + RMSNorm; QK-norm caps logits as the projections grow |
| The FFN learned to gate | `05_ffn_gate.py` | The 8/3 width rule; GELU FFN vs SwiGLU at equal parameters on a product and a control target |
| The KV cache became the constraint | `06_kv_cache.py` | Llama 3 and DeepSeek-V3 cache bytes, decode intensity, the formula against real cache tensors, MLA absorption and why RoPE breaks it |
| The KV cache (bandwidth) | `benchmarks/gpu_bench.py` | Decode time and achieved bandwidth as n_kv shrinks |
| The mask grew holes | `07_sliding_window.py` | A ring-buffer cache equals sliding-window attention; Olmo 3 bytes; receptive field per interleaving |
| One FFN became hundreds | `08_moe.py` | C(256, 8); total and active parameters; routing with a head start, with and without balancing |
| Past softmax: the 2026 hybrids | `09_beyond_softmax.py` | Parallel = recurrent linear attention; recall from a 64 x 64 state, additive vs delta rule; the decay gate |
| Small changes that aren't small | `10_small_changes.py`, `benchmarks/gpu_bench.py` | Logits memory; chunked cross-entropy; weight tying share; attention sinks under eviction |
| Build a model | `11_spec_calculator.py` | The Figure 10 calculator for 7 presets, checked against exact counts from the real configs |

## Results

Raw output is in `results/`, under the name of the script that wrote it. Each file starts with its hardware line
and the command.

Tests on AMD EPYC 7R13, 8 threads, torch 2.15.0: 52 offline tests pass, 1 `network` test passes, 4 `gpu` tests
pass on an NVIDIA H100 80GB HBM3 (`results/gpu_tests.txt`).

### Arithmetic from the post, recomputed (no hardware dependence)

| Claim in the post | Recomputed |
|---|---|
| Llama 3 8B shape with full MHA: 512 KiB per token, 64 GiB at 128k | 512 KiB, 64.0 GiB (GQA as shipped: 128 KiB, 16 GiB) |
| Decode intensity is n_q / n_kv, independent of d_h and s | 4.00 FLOPs/byte for 32 over 8 heads at every d_h and s tried |
| H100 ridge point near 295 FLOPs/byte | 295 |
| DeepSeek-V3 MLA: 70,272 B/token vs 4.77 MiB, 71x, 8.6 GiB vs 610 GiB at 128k | 70,272 B, 4.77 MiB, 71.1x, 8.6 vs 610 GiB |
| Olmo 3 32B at 65,536 tokens: 16 GiB full, 4.75 GiB with 3:1 SWA, 3.4x | 16.00, 4.75 GiB, 3.37x |
| Llama 2: 8d/3 = 10,923, shipped 11,008 | 10,922.7, next multiple of 256 = 11,008 |
| C(8, 2) = 28, C(256, 8) about 4.1e14 | 28, 4.10e14 |
| Qwen3 logits for B 8, S 8192, fp32: about 37 GiB | 37.1 GiB |
| Figure 10 totals for 7 presets | within 0.2% of the exact count from each model's config.json (GPT-2 XL -0.17%, the rest under 0.03%) |
| MoE table, active parameters: Qwen3 22B, gpt-oss 5.1B, DeepSeek-V3 37B | 21.5B, 5.1B, 36.5B |

The C4 `KVCache` held exactly the bytes the formula predicts (589,824 for 4 layers, 2 KV heads, d_h 32, 96 tokens,
batch 3, fp32).

### Mechanisms, AMD EPYC 7R13, 8 threads, torch 2.15.0

| Experiment | Result |
|---|---|
| Pre-LN stream at init, 24 layers: fit of the squared norm to a + b l | R^2 = 0.992, so the norm grows like sqrt(l); 8.0x from layer 1 to 24 |
| Post-LN stream at init | pinned at sqrt(d) = 11.3 after every layer |
| Last-block FFN gradient, depth 6 / 24 / 48 | Post-LN 0.0101 / 0.0104 / 0.0101 (flat); Pre-LN 0.144 / 0.057 / 0.036 (shrinks with depth) |
| 12-layer LM, lr 3e-3, 300 steps, 2 seeds: validation loss (nats) | Post-LN 1.95 / 1.92; Post-LN + 100-step warmup 1.97 / 1.98; Pre-LN 2.20 / 2.25; sandwich 2.23 / 2.26; post-branch 2.87 / 2.87. No run spiked |
| 24-layer LM, lr 1e-2, 200 steps, 1 seed: validation loss | Post-LN 3.33; Post-LN + warmup 3.34 (1 spike); Pre-LN 2.98 |
| RoPE logit for offset 3 at positions (3,0), (103,100), (4003,4000) | +1.908099, +1.908090, +1.908000 (float32 rounding) |
| RoPE pairs with wavelength over 8k tokens, d_h 128 | base 1e4: 14 of 64; base 5e5: 29; base 1e6: 30 |
| NoPE, one random head: shuffle error / probe R^2 for log position | bidirectional 1.8e-15 / -0.15; causal 4.3 / 0.47 |
| Attention logits when W_q, W_k grow 16x | max logit 4.5 -> 1147, entropy 4.09 -> 0.01 nats; with QK-norm 3.8 and 4.08 throughout |
| Test MSE / Var(y), product target, matched parameters | GELU FFN 0.0064, SwiGLU 0.0001 |
| Same, control target sin(a) + cos(b) | GELU FFN 0.107, SwiGLU 0.237 |
| Ring-buffer sliding-window decode vs masked attention, 512 tokens | max difference 2.7e-15; cache 16 KiB for w = 16 against 512 KiB full |
| Receptive field after 12 local layers, window 32 | 372 tokens = 12 x 31; one global layer reaches all 1023 |
| MoE, expert 0 given a +4 head start: max / mean load | no balancing 4.16 (0.5 dead experts on average), aux loss 2.45, aux-free bias 1.08 |
| MoE test MSE with the head start | none 0.0554, aux 0.0528, bias 0.0526 |
| Recall from a 64 x 64 state, 32 pairs, correlated keys (cosine) | additive 0.33, delta rule 0.51, softmax over full cache 1.00 |
| Rewrite 16 keys with new values: cosine to the new value | additive 0.62, delta rule 0.91 |
| Chunked cross-entropy (chunk 256) vs full | loss difference 9.5e-7, gradient difference 8e-10 |
| Evicting token 0 from a head that dumps on it: output norm | 0.021 -> 1.289 with plain softmax; 0.021 -> 0.021 with a sink logit |
| LayerNorm vs RMSNorm, (16384, 4096) fp32, CPU | F.layer_norm 66 ms, F.rms_norm 139 ms; eager two reductions 164 ms, one reduction 127 ms |

The CPU norm timing points the other way from the post for the PyTorch kernels: the CPU `layer_norm` kernel is
fused and `rms_norm` is not. The eager code, which does the reductions as written, is faster with one. The post's
claim is about GPUs and tensor parallelism. The GPU table below is the relevant one.

### NVIDIA H100 80GB HBM3

One H100, torch 2.15.0, CUDA 13.2. The job ran alone on its GPU. Full output: `results/gpu_bench.txt`.

Decode attention, batch 8, 32 query heads, d_h 128, a 32,768-token bf16 cache, median of 50:

| n_kv | KV read | SDPA time | SDPA GB/s | grouped-einsum time | grouped GB/s | grouped TFLOP/s |
|---|---|---|---|---|---|---|
| 32 (MHA) | 4096 MiB | 1.371 ms | 3133 | 1.503 ms | 2857 | 2.9 |
| 8 (Llama 3) | 1024 MiB | 0.376 ms | 2855 | 0.469 ms | 2289 | 9.2 |
| 4 | 512 MiB | 0.209 ms | 2571 | 0.303 ms | 1772 | 14.2 |
| 1 (MQA) | 128 MiB | 0.092 ms | 1462 | 0.217 ms | 618 | 19.8 |

Decode attention is memory-bound, as the post says. With 8 or 32 KV heads the kernels read at 2.3 to 3.1 TB/s,
against a 3.35 TB/s spec, and reach at most 20 TFLOP/s of about 989. Time falls with the bytes read: MHA to GQA-8
is 3.6x faster with SDPA for 4x fewer bytes. It does not fall all the way. MHA to MQA is 15x faster for 32x fewer
bytes, because a 128 MiB read is too small to keep the memory system busy (1.5 TB/s).

Normalization, (65,536, 4,096) bf16, forward only:

| op | time | effective GB/s |
|---|---|---|
| `F.layer_norm` | 0.534 ms | 2009 |
| `F.rms_norm` | 0.539 ms | 1992 |
| two reductions, eager | 2.131 ms | 504 |
| one reduction, eager | 1.232 ms | 872 |

The post's argument that RMSNorm halves the reductions is not visible in PyTorch's fused kernels. Both run at the
same speed, limited by one read and one write. The eager gap (1.7x) is real but mostly counts memory passes: the
two-reduction code writes the centered tensor and reads it again. The effect the post describes, a synchronization
point per reduction and a collective under tensor parallelism, was not measured: this is one GPU.

Cross-entropy, 8,192 tokens, d 4,096, V 151,936, fp32 logits (one logits tensor is 4.64 GiB):

| loss | value | peak extra memory (forward + backward) |
|---|---|---|
| full | 12.7710 | 13.91 GiB |
| chunked, 1,024 tokens | 12.7710 | 3.55 GiB |
| chunked, 256 tokens | 12.7710 | 3.54 GiB |

The weight and hidden gradients account for 1.22 GiB of the 3.5 GiB. The rest was not broken down. Chunking by 256
instead of 1,024 changes nothing at this size, so the floor is not the logits.

## What these results do and do not show

- **The warmup claim was not reproduced.** The post says Post-LN needs a warmup to train at all. At 12 and 24
  layers, with 300 and 200 steps, Post-LN trained without warmup and without a spike, and at 12 layers it reached
  a lower loss than Pre-LN. Short runs favour Post-LN, because Pre-LN's small early updates make it slower to
  start. The measured part is the gradient: the last-block gradient stays flat with depth in Post-LN and shrinks
  in Pre-LN (table above), which is the mechanism behind the claim. The instability appears at greater depth and
  learning rates and longer runs than a CPU allows here. The 24-layer run used one seed.
- The warmup timings in `results/02_warmup.txt` ran on a shared machine and vary by 8x between rows (95 s to
  845 s per run) with no change in the work done. Do not read them as speeds.

- Every training run here is tiny: about 2.6M parameters for 300 steps, or a single MoE layer on a toy regression.
  They show each mechanism happening. They do not measure the size of any effect at the scale the post discusses.
- The post's quality claims about large models (GQA costs almost no quality, many small experts beat few large
  ones, linear hybrids match full attention, dropout hurts single-epoch training, wider vs deeper) are quoted from
  papers. None was reproduced.
- The SwiGLU result is a toy. The gate wins on a multiplicative target and loses on an additive one. That matches
  the post's caution that GLU's gain in language models has no settled explanation; it does not explain it.
- The MoE head-start run shows rich-get-richer routing in a single layer. Without the head start, this toy did not
  collapse on its own. Expert parallelism and all-to-all traffic were not measured.
- MLA was not trained or benchmarked. Absorption and the RoPE obstruction are checked on one token; the full layer
  with the decoupled RoPE key is in C3.
- The NoPE probe is a linear probe on one random head. It shows that position is present in the output, not that a
  trained NoPE layer uses it.
- Model sizes and context numbers for real models are arithmetic from published configs, not measurements.
