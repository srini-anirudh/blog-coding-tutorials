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
| `python -m pytest c03-attention -m gpu -rs` | The CUDA kernels against naive. GPU-only | `results/gpu_tests.txt` |
| `python c03-attention/benchmarks/bench_backends.py --flash` | The backend table on CUDA, with the memory-efficient and flash kernels. GPU-only | `results/gpu_bench_backends.txt` |
| `python c03-attention/benchmarks/bench_zoo.py --device cuda --out c03-attention/results/gpu_bench_zoo.json` | The zoo on CUDA, with a GQA flash row and cuDNN attention off. GPU-only | `results/gpu_bench_zoo.txt`, `results/gpu_bench_zoo.json` |
| `python c03-attention/benchmarks/bench_zoo.py --device cuda --cudnn --out /dev/null` | The same with PyTorch's default kernel choice, cuDNN attention on. GPU-only | `results/gpu_bench_zoo_cudnn.txt` |

The GPU commands are collected in the GPU queue script the coordinator runs. Run each with one CUDA GPU visible.

At N = 16384 the naive, MLA and linear rows of the zoo need 18 to 25 GB of memory. On a smaller machine pass shorter lengths, for example `--lengths 512 1024 2048`.

## CPU results

Hardware for every number in this section: **AMD EPYC 7R13, 8 threads, torch 2.15.0, CPU only, fp32**. The machine is shared with other jobs, so timings carry noise, most of all at N = 8192 and above, where each cell is one measured call. Each saved output starts by naming its hardware.

### Backends: one causal attention call

H = 4, d = 32. Milliseconds per call / peak extra MB.

| Backend | N = 1024 | N = 4096 | N = 16384 |
|---|---|---|---|
| `naive` | 9.9 / 69 | 449.7 / 814 | 7201.7 / 12528 |
| `sdpa_math` | 27.8 / 39 | 282.3 / 676 | 4154.4 / 10257 |
| `sdpa` | 2.4 / 8 | 29.2 / 10 | 300.0 / 12 |
| `sdpa_flash` (CPU kernel) | 2.8 / 8 | 24.7 / 9 | 282.4 / 12 |
| `tiled` (Python, for learning) | 37.4 / 3 | 512.2 / 6 | 7771.5 / 9 |

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

Three `Linear` layers took 38.33 ms per call and one fused `Linear` took 39.21 ms, on 8 x 1024 x 512 inputs. On this CPU the fused layer is usually slightly slower, and the spread between runs exceeds the gap. The output is the same. The usual argument for fusing is fewer kernel launches, and that was not measured here.

## GPU results

Hardware for every number in this section: **NVIDIA H100 80GB HBM3, torch 2.15.0 built for CUDA 13.2, bf16**, one GPU, through a lock so no other job shared it. The FlashAttention kernels accept only fp16 and bf16, so these runs use bf16 and the CPU runs use fp32. Do not compare the two sections cell by cell.

The `flash-attn` package is not installed. So the `flash` backend runs PyTorch's FlashAttention-2 through SDPA, the same kernel as `sdpa_flash`, and windows and soft-capping on `flash` raise. Tri Dao's kernels are not measured here.

### GPU tests

`pytest c03-attention -m gpu`: 7 passed, 1 skipped. The skip is the window-and-decode test, which needs the `flash-attn` package. The `flash` backend matches naive in bf16 for causal and bidirectional attention, with Hkv = 8 and Hkv = 2, and decodes one token against a 300-token cache. The pinned FlashAttention-2 and memory-efficient kernels match naive for causal MHA.

### Backends: one causal attention call

H = 4, d = 32, B = 1. Milliseconds per call / peak extra MB. The table prints one decimal, so cells under 1 ms are rough.

| Backend | N = 1024 | N = 4096 | N = 16384 |
|---|---|---|---|
| `naive` | 0.4 / 49 | 1.5 / 656 | 21.0 / 10496 |
| `sdpa_math` | 0.5 / 62 | 1.7 / 648 | 22.2 / 10272 |
| `sdpa_efficient` | 0.1 / 0 | 0.2 / 1 | 1.6 / 4 |
| `sdpa_flash` (FlashAttention-2) | 0.1 / 2 | 0.1 / 1 | 0.6 / 4 |
| `flash` (FlashAttention-2 through our wrapper) | 0.4 / 2 | 0.4 / 1 | 1.0 / 4 |
| `tiled` (Python, for learning) | 65.8 / 33 | 970.0 / 33 | 14758.9 / 36 |

At N = 16384, FlashAttention-2 is 35 times faster than `naive` and needs about 2600 times less extra memory. The `flash` row runs the same kernel as `sdpa_flash` but is slower at every length. Our wrapper adds Python work per call, including a failed `import flash_attn`, which takes about 0.09 ms on the CPU machine. That does not explain the whole gap, and we did not look further. `tiled` is slower on the GPU than on the CPU, because it launches many tiny kernels from Python.

### The zoo at N = 16384

One layer, D = 256, Hq = 8, dh = 32, B = 1. cuDNN attention is off: see the next section. The other lengths are in `results/gpu_bench_zoo.txt`.

| Layer | Forward ms | Backward ms | Peak MB | Prefill ms | Decode ms/token | Cache MB |
|---|---|---|---|---|---|---|
| MHA | 2.2 | 2.5 | 83 | 1.6 | 2.64 | 16.02 |
| GQA Hkv=2 | 2.1 | 2.8 | 75 | 2.1 | 1.47 | 4.00 |
| MQA | 2.7 | 3.3 | 71 | 4.2 | 2.42 | 2.00 |
| MLA r=64 | 5.2 | 8.9 | 124 | 5.0 | 2.80 | 2.25 |
| Local, mask | 9.0 | 16.2 | 791 | 9.2 | 1.59 | 4.00 |
| Local, chunked | 22.3 | 21.3 | 95 | 16.0 | 1.07 | 4.00 |
| Linear | 12.3 | 18.3 | 8792 | 3467.8 | 0.67 | 0.02 |
| GQA, naive backend | 41.9 | 56.1 | 33050 | 42.0 | 3.22 | 4.00 |
| GQA, flash backend | 2.5 | 5.8 | 75 | 3.4 | 1.57 | 4.00 |

How to read this table:

- This layer is too small to keep an H100 busy. Most rows take 1 to 3 ms at every length from 512 to 16384, so they measure kernel launches and Python, not attention. The rows are not a ranking of the architectures. Each cell at N >= 8192 is one call after a warm-up, so expect noise of a few ms.
- What does show through is memory. The naive backend holds the score matrix and peaks at 33050 MB. The fused kernels stay under 125 MB.
- The CPU lessons do not all carry over. In the CPU zoo the chunked window layer was 5 times faster than MHA. Here it is 10 times slower, because its Python loop issues 64 small calls where FlashAttention-2 issues one. A real speed-up from a window needs a kernel that skips the tiles, such as the `flash-attn` window, which is not installed.
- The CPU backward gap with `enable_gqa` does not appear: GQA and MQA backward are within about 1 ms of MHA. At this size that is all we can say.
- Linear prefill takes 3.5 s, because prefill with a cache runs the recurrence one token at a time in Python.
- Cache MB is half the CPU value, because bf16 uses 2 bytes per number.

### cuDNN attention and new shapes

The first GPU zoo run used PyTorch's default kernel choice. On this H100 that picked cuDNN attention, and the default `sdpa` rows reported 61 to 74 ms per prefill and 48 to 91 ms per decoded token at every length. The chunked window, which makes many calls per prefill, took 141 to 224 ms. The first prefill at N = 512 took 557 to 1131 ms. A direct probe of the SDPA call, with Nq = 1 and a cache of about 2000 keys, found the cause:

| Kernel | Decode, new cache length each step | Decode, the same shape again |
|---|---|---|
| Default choice (cuDNN) | 78.84 ms | 0.144 ms |
| Default, cuDNN excluded | 0.05 ms | 0.053 ms |
| FlashAttention-2, pinned | 0.05 ms | 0.043 ms |
| Memory-efficient, pinned | 0.17 ms | 0.166 ms |

These rows are for MHA (Hkv = 8). With `enable_gqa`, the default choice took 61.61 ms per new shape. cuDNN pays about 60 to 80 ms the first time it sees each shape and is fast after that. Prefill and every decode step bring a new key length, so the zoo timed that setup and not attention. The fix is in `bench_zoo.py`: on CUDA it switches cuDNN attention off unless you pass `--cudnn`, and SDPA then picks FlashAttention-2 or the memory-efficient kernel. Training forward and backward were unaffected, because a warm-up call ran on the same shape first. The first run is kept in `results/gpu_bench_zoo_cudnn.txt`. It was made before the flag existed, on the same code path that `--cudnn` takes now.

The probe script is not in this folder. The per-shape cost is measured, but its cause inside cuDNN, a setup step cached by shape, is an inference: we did not read the cuDNN or PyTorch source. A server that pads cache lengths to a few fixed sizes would pay the cost only a few times.
