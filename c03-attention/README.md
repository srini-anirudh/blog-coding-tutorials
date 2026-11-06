# C3: Build Attention From Scratch

Part of [blog-coding-tutorials](https://github.com/srini-anirudh/blog-coding-tutorials). This folder builds the package `attnzoo`: attention from one head up to GQA, MLA, online softmax and multimodal fusion. Every implementation is checked against a trusted one.

## What you build

| Module | What it holds |
|---|---|
| `core.py` | Scaled dot-product attention with every step visible, and one self-attention head |
| `masks.py`, `biases.py` | Bidirectional, causal, padding, prefix, document, multimodal, sliding-window, local + global and block-sparse masks. ALiBi and relative position biases |
| `heads.py`, `mha.py` | Head reshapes, MHA, fused QKV projection, MQA |
| `gqa.py`, `config.py` | One module for MHA, MQA and GQA, driven by `AttentionConfig` |
| `rope.py`, `kv_cache.py` | RoPE on Q and K. Dynamic and preallocated caches, with the memory formula |
| `cross.py`, `resampler.py`, `fusion.py` | Cross-attention, Q-Former, Perceiver resampler, early, cross and gated fusion |
| `vision.py`, `pos2d.py` | ViT attention, windows and shifted windows, 2D positions, 2D RoPE |
| `linear.py` | Linear attention, its recurrence, and Performer features (advanced, research) |
| `mla.py` | Multi-head latent attention in four steps, plus the absorbed form |
| `stability.py`, `variants.py` | QK-norm and soft-capping. Gated, cosine, differential and talking-heads attention (research) |
| `attention_naive.py`, `attention_sdpa.py`, `attention_tiled.py`, `attention_window.py`, `attention_flash.py` | Execution backends |
| `backends.py`, `library.py` | `attend`, `build_attention` and `make_cache`: architecture kept apart from backend |

### Backends

| Backend | What runs | Where |
|---|---|---|
| `naive` | The reference. Holds the whole score matrix | Anywhere |
| `sdpa` | Whatever kernel PyTorch picks | Anywhere |
| `sdpa_math` | PyTorch's unfused path | Anywhere |
| `sdpa_flash` | On CPU, PyTorch's CPU flash kernel. On CUDA, FlashAttention-2 | CPU or CUDA |
| `sdpa_efficient` | PyTorch's memory-efficient kernel | CUDA only |
| `tiled` | Online softmax in plain Python, for learning. It is not a fast kernel | Anywhere |
| `window_chunked` | Sliding window, truly O(N W) | Anywhere |
| `flash` | The `flash-attn` package, or FlashAttention-2 through SDPA when the package is missing | CUDA only |

The CPU flash kernel and the CUDA kernels are different programs. A CPU number for `sdpa_flash` says nothing about a GPU.

## How to run

Run everything from the repository root. Keep the thread cap in `attnzoo/device.py` in mind: scripts and benchmarks use at most 8 threads, and tests use 2.

### Tests

```bash
python -m pytest c03-attention -m "not slow and not gpu and not network"   # offline, about 20 s
python -m pytest c03-attention -m gpu -rs                                   # needs a CUDA GPU, otherwise they skip
```

The suite covers shapes, causality, manual MHA against SDPA and against `torch.nn.MultiheadAttention`, GQA against MHA and MQA, cached decoding against a full forward pass, gradient flow, fp32 against bf16, and our GQA against the Hugging Face `LlamaAttention` with the same weights. This folder has no `network` or `slow` tests. The saved output is `results/tests.txt`.

One test skips on purpose: MLA on `sdpa_flash`. Flash kernels need q, k and v of the same width, and MLA's values are narrower than its keys.

### Demos

Each prints what it builds. The saved output is in `results/<name>.txt`.

| Command | What it shows |
|---|---|
| `python c03-attention/scripts/shapes.py` | Every tensor shape: one head, many heads, fused QKV, cross-attention |
| `python c03-attention/scripts/show_masks.py` | Every mask as an ASCII picture |
| `python c03-attention/scripts/compare_heads.py` | MHA, GQA and MQA side by side, and separate against fused projections |
| `python c03-attention/scripts/kv_cache_demo.py` | RoPE tables, prefill and decode, the preallocated cache, position offsets |
| `python c03-attention/scripts/mla_demo.py` | MLA in four steps, cached decoding, cache size against parameters |
| `python c03-attention/scripts/linear_demo.py` | The reordered matmuls, the recurrence, Performer features |
| `python c03-attention/scripts/online_softmax_demo.py` | The running `m` and `l` through two tiles, then tiled against naive |
| `python c03-attention/scripts/stability_demo.py` | Dropout, ALiBi, QK-norm, soft-capping, research variants |
| `python c03-attention/scripts/vision_demo.py` | Patch attention maps, windows, shifted windows, 2D positions |
| `python c03-attention/scripts/resampler_demo.py` | Q-Former, Perceiver, the three fusion patterns |
| `python c03-attention/scripts/library_demo.py` | One config, any architecture, any backend |
| `python c03-attention/scripts/break_it.py` | Six ways to get attention wrong, with the real symptoms |
| `python c03-attention/scripts/export_figure_data.py` | Writes `results/figure_data.json`, the tensors the post's figures draw |

### Benchmarks

Each cell of the backend and zoo tables runs in a fresh process. A warm-up call runs first and is not counted.

| Command | What it measures | Saved as |
|---|---|---|
| `python c03-attention/benchmarks/bench_kv_memory.py` | The KV memory formula against real cache tensors | `results/bench_kv_memory.txt` |
| `python c03-attention/benchmarks/bench_window.py` | A sliding window through a mask and through the chunked kernel, and block-sparse masks | `results/bench_window.txt` |
| `python c03-attention/benchmarks/bench_backends.py` | Latency, peak memory and scaling for each CPU backend | `results/bench_backends.txt` |
| `python c03-attention/benchmarks/bench_zoo.py` | Training, prefill and decode for each architecture | `results/bench_zoo.txt`, `results/bench_zoo.json` |
| `python c03-attention/benchmarks/bench_backward.py` | Backward time with `enable_gqa` against `repeat_kv`, across KV heads, threads and batch sizes. About 2 minutes | `results/bench_backward.txt` |
| `python c03-attention/benchmarks/bench_backends.py --flash` | The same table on CUDA, with the memory-efficient and flash kernels. GPU-only | Not yet measured |
| `python c03-attention/benchmarks/bench_zoo.py --device cuda --out results-file.json` | The zoo on CUDA, with a GQA flash row. GPU-only | Not yet measured |

At N = 16384 the naive, MLA and linear rows of the zoo need 18 to 25 GB of memory. On a smaller machine pass shorter lengths, for example `--lengths 512 1024 2048`.

## Results

Hardware for every number below: **AMD EPYC 7R13, 8 threads, torch 2.15.0, CPU only, fp32**. The machine is shared with other jobs, so timings carry noise, most of all at N = 8192 and above, where each cell is one measured call. Each saved output starts by naming its hardware.

### Backends: one causal attention call

H = 4, d = 32. Milliseconds per call / peak extra MB.

| Backend | N = 1024 | N = 4096 | N = 16384 |
|---|---|---|---|
| `naive` | 9.9 / 69 | 449.7 / 814 | 7201.7 / 12528 |
| `sdpa_math` | 27.8 / 39 | 282.3 / 676 | 4154.4 / 10257 |
| `sdpa` | 2.4 / 8 | 29.2 / 10 | 300.0 / 12 |
| `sdpa_flash` (CPU kernel) | 2.8 / 8 | 24.7 / 9 | 282.4 / 12 |
| `tiled` (Python, for learning) | 37.4 / 3 | 512.2 / 6 | 7771.5 / 9 |
| `sdpa_efficient` (CUDA) | Not yet measured | Not yet measured | Not yet measured |
| `flash` (CUDA) | Not yet measured | Not yet measured | Not yet measured |

Tiling keeps peak memory near flat as N grows, even in slow Python. The materialising paths need roughly 850 to 1000 times more memory than `sdpa` at N = 16384.

### A sparse mask on a dense tensor is still dense

Sliding window W = 256, H = 4, d = 32. Milliseconds per call.

| N | Full causal | Mask + naive | Mask + SDPA | Chunked |
|---|---|---|---|---|
| 1024 | 3.8 | 15.8 | 3.6 | 2.3 |
| 4096 | 18.0 | 361.9 | 57.9 | 7.9 |
| 16384 | 258.6 | 5984.2 | 958.7 | 31.5 |

Block-sparse masks at N = 4096 show the same thing. Going from density 0.508 to 0.031 moves `naive` from 391.4 ms to 403.6 ms and `sdpa` from 47.4 ms to 45.6 ms. The time does not fall.

### The zoo at N = 16384

One layer, D = 256, Hq = 8, dh = 32. The other lengths are in `results/bench_zoo.txt`.

| Layer | Forward ms | Backward ms | Peak MB | Prefill ms | Tokens/s | Decode ms/token | Cache MB |
|---|---|---|---|---|---|---|---|
| MHA | 515.0 | 1333.3 | 325 | 507.6 | 32,279 | 2.13 | 32.03 |
| GQA Hkv=2 | 447.0 | 3763.2 | 212 | 680.2 | 24,086 | 2.85 | 8.01 |
| MQA | 380.1 | 6680.5 | 110 | 406.0 | 40,353 | 1.10 | 4.00 |
| MLA r=64 | 5632.0 | 7372.2 | 24757 | 8077.4 | 2,028 | 12.16 | 4.50 |
| Local, mask | 1428.7 | 8342.9 | 1334 | 1808.3 | 9,060 | 1.81 | 8.01 |
| Local, chunked | 102.6 | 469.7 | 328 | 80.1 | 204,642 | 1.44 | 8.01 |
| Linear | 4758.4 | 9864.6 | 17665 | 1120.2 | 14,625 | 0.23 | 0.03 |
| GQA, naive backend | 9868.3 | 13914.6 | 25206 | 12260.0 | 1,336 | 4.09 | 8.01 |
| GQA, flash backend (CUDA) | Not yet measured | | | | | | |

Three things in this table need care.

- MLA pays for its small cache here. Its q and k are wider than its v, so the CPU flash kernel refuses it and SDPA falls back to the math path. That is why its peak memory matches the naive row.
- Backward time rises as KV heads fall, at every length: MHA, then GQA, then MQA. Forward time does not. This table uses B = 1, and the gap comes from `enable_gqa=True` on the CPU kernel. The next section measures it.
- Linear attention holds a constant 0.03 MB of decode state. Its training path is the parallel form, which is O(N^2) again.

### Backward time and `enable_gqa`

`bench_zoo.py` and the tables here use B = 1 unless a batch size is stated. `bench_backward.py` compares the flag with `repeat_kv`, which copies K and V up to Hq heads first. Cells are the median of 5 calls after a warm-up, and the full file gives the fastest and slowest call. On the same weights, the two paths give the same output and parameter gradients within 8.7e-11, at N = 1024.

One attnzoo layer, B = 1, 8 threads, N = 8192, backward ms:

| Layer | With the flag | With `repeat_kv` |
|---|---|---|
| MHA | 288.2 (no flag needed) | |
| GQA Hkv=2 | 954.1 | 271.2 |
| MQA | 1874.2 | 270.0 |

The SDPA kernel alone, B = 1, N = 4096, backward ms:

| Path | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---|---|---|---|
| Hkv=8, no flag | 474.0 | 242.9 | 128.1 | 78.7 |
| Hkv=4, flag | 460.3 | 232.1 | 134.9 | 124.7 |
| Hkv=2, flag | 456.9 | 246.1 | 242.6 | 238.7 |
| Hkv=1, flag | 462.2 | 382.6 | 651.8 | 514.0 |
| Hkv=1, `repeat_kv` | 459.9 | 243.6 | 131.9 | 80.5 |

At 1 thread every path matches. With the flag, the speed-up stops once the thread count passes Hkv, and with one KV head more threads do not help.

Flag time divided by `repeat_kv` time, one layer, 8 threads, N = 2048:

| Batch size | GQA Hkv=2 | MQA |
|---|---|---|
| 1 | 2.44 | 4.43 |
| 2 | 1.58 | 2.22 |
| 4 | 0.93 | 1.72 |
| 8 | 1.01 | 1.13 |

The gap closes once B x Hkv reaches the thread count. An earlier run of the same script agreed: 2.69 and 5.37 at B = 1, 0.95 and 1.66 at B = 4, 0.97 and 0.98 at B = 8.

These results fit a backward kernel that splits its work over B x Hkv and not over B x Hq. That mechanism is an inference from the timings. We did not read the kernel source. The numbers cover this CPU kernel in fp32 only and say nothing about a GPU. `repeat_kv` also gives up the flag's no-copy K and V, and this script does not measure that memory cost.

### KV memory formula

`M_KV = 2 L B S Hkv dh b`, with L = 2, B = 1, D = 1024, Hq = 16, dh = 64, fp32. The formula equals the bytes in the real cache tensors in every measured row, in fp32 and bf16.

| Layer | S = 1024 | S = 16384 |
|---|---|---|
| MHA | 16,777,216 | 268,435,456 |
| GQA Hkv=4 | 4,194,304 | 67,108,864 |
| MQA | 1,048,576 | 16,777,216 |
| MLA r=128, dr=16 | 1,179,648 | Not run |

MLA uses its own formula, `L B S (r + dr) b`. Part 2 of `results/bench_kv_memory.txt` applies the formulas to L = 32 configurations. Those numbers are computed, not measured.

### Equivalence checks

| Check | Max absolute difference | Saved in |
|---|---|---|
| Tiled online softmax against naive, N = 512 | 7.15e-07 | `online_softmax_demo.txt` |
| Unified GQA against separate MHA | 1.64e-07 | `compare_heads.txt` |
| Unified GQA against separate MQA | 1.79e-07 | `compare_heads.txt` |
| Cached decode against full forward, GQA | 1.0e-07 | `kv_cache_demo.txt` |
| MLA absorbed form, cached, against full forward | 1.9e-07 | `mla_demo.txt` |
| Linear recurrence against parallel form | 3.6e-07 | `linear_demo.txt` |

### Fused QKV projection

Three `Linear` layers took 38.33 ms per call and one fused `Linear` took 39.21 ms, on 8 x 1024 x 512 inputs. The fused layer was not faster in any of the five runs we made on this CPU. The output is the same. The usual argument for fusing is fewer kernel launches, and that was not measured here.

## GPU-only: not yet measured

These need a CUDA GPU and have no numbers yet.

- The `flash` backend, through `flash-attn` or through PyTorch's FlashAttention-2.
- The `sdpa_efficient` backend.
- The 8 tests in `tests/test_gpu.py`. They skip without a GPU and have never run on one.

The CUDA benchmarks run in bf16, because the FlashAttention kernels accept only fp16 and bf16. Do not compare them directly with the fp32 CPU tables above.

Without the `flash-attn` package, the `flash` and `sdpa_flash` rows run the same kernel, and windows and soft-capping raise. `bench_backends.py --flash` prints which route ran.
