# Lab: How Models Know Where They Are

A code companion for the theory post on positional encoding. Each script builds one of the post's mechanisms in a
few dozen lines of PyTorch, runs it, and prints what happens. Everything runs on a CPU.

Post: https://srini-anirudh.github.io/blogs/how-models-know-where-they-are/

The lab does not rebuild what the C3 tutorial already has. Half-split RoPE (`rotate_half`), ALiBi, axial 2D RoPE
and the 2D sin-cos table are imported from [`c03-attention/attnzoo`](../../c03-attention/). This lab adds the
parts the post is about: what each scheme can and cannot express, what happens past the training length, and the
bookkeeping in vision-language models.

## Layout

```
labs/how-models-know-where-they-are/
├── posenc/
│   ├── c03.py         # imports RoPE, ALiBi, 2D RoPE and 2D sin-cos from C3
│   ├── sinusoidal.py  # the frequency code and the fixed rotation R_k with PE(p+k) = R_k PE(p)
│   ├── rope.py        # half-split and interleaved RoPE, the permutation between them, decay envelope, precision
│   ├── extend.py      # Position Interpolation, NTK-aware scaling, YaRN
│   ├── tinylm.py      # a 2-layer decoder with a switchable scheme: none, learned, sinusoidal, rope, alibi
│   ├── tasks.py       # synthetic data: predict-your-position, induction copy
│   ├── vision2d.py    # raster ids, resizing a learned position table with a cls row
│   ├── mrope.py       # M-RoPE ids, chunked and interleaved layouts, absolute-time ids, footprint
│   ├── fourier.py     # random Fourier features, coordinate MLP, SAM-style kernel
│   └── report.py      # writes results/<script>.txt with a hardware line and the command
├── scripts/           # 01 to 06, one per group of post sections
├── results/           # real outputs of every script
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Nothing is downloaded. Two checks call code inside the
installed `transformers` package (YaRN frequencies, Qwen2-VL and Qwen3-VL M-RoPE), with no model weights.

## Run everything

```bash
# Tests, about 20 s.
python -m pytest labs/how-models-know-where-they-are -m "not slow and not gpu and not network"

# Scripts. 02, 04 and 05 take seconds. 01 and 06 take a few minutes. 03 trains 10 small models.
python labs/how-models-know-where-they-are/scripts/01_symmetry.py
python labs/how-models-know-where-they-are/scripts/02_sinusoidal_rope.py
python labs/how-models-know-where-they-are/scripts/03_extend.py              # all five schemes, seeds 0 and 1
python labs/how-models-know-where-they-are/scripts/04_vision_2d.py
python labs/how-models-know-where-they-are/scripts/05_mrope.py
python labs/how-models-know-where-they-are/scripts/06_fourier.py

# 03 split into one process per scheme, then merged (how results/ was made):
for pe in rope alibi none sinusoidal learned; do
  OMP_NUM_THREADS=4 python labs/how-models-know-where-they-are/scripts/03_extend.py --pe $pe --name 03_extend_$pe &
done; wait
python labs/how-models-know-where-they-are/scripts/03_extend.py \
  --merge 03_extend_rope,03_extend_alibi,03_extend_none,03_extend_sinusoidal,03_extend_learned
```

## Which script covers which section

| Post section | Script | What it shows |
|---|---|---|
| 01 The symmetry to break | `01_symmetry.py` | attention without position is permutation equivariant; a mask or RoPE breaks it |
| 02 Absolute: a table of addresses | `03_extend.py`, `tinylm.py` | a learned table has no row past its length |
| 03 Sinusoidal: a frequency code | `02_sinusoidal_rope.py` | PE(p+k) = R_k PE(p); PE(p).PE(p+k) depends on k only |
| 04 Relative: distances, not addresses | `03_extend.py` | ALiBi (from C3) trained short, run long |
| 05 RoPE: rotate, don't add | `02_sinusoidal_rope.py` | relative scores, norm preservation, the two pairing conventions, the decay envelope |
| 06 Stretching RoPE past its training length | `03_extend.py`, `02_sinusoidal_rope.py` | PI, NTK and YaRN speeds; trained models at 4x length; planes that never finish a turn |
| 07 NoPE, and the hybrids that won | `01_symmetry.py`, `03_extend.py` | the causal mask leaks position; NoPE trained short, run long |
| 08 Two dimensions | `04_vision_2d.py` | raster ids versus axial 2D RoPE; resizing a table with a cls row |
| 09 Fourier features, NeRF, and SAM | `06_fourier.py` | sigma too small blurs, too large overfits; a shared B gives a kernel peaked at the click |
| 10 M-RoPE: three axes at once | `05_mrope.py` | ids match Qwen2-VL's code; chunked versus interleaved wavelengths; absolute time; max + 1 |
| 11 Position bookkeeping in VLM pipelines | `05_mrope.py` | positional footprint of one image under flat ids, M-RoPE and V2PE |
| 13 Nuances: precision, conventions | `02_sinusoidal_rope.py` | bf16 versus fp32 angle error; mismatched conventions |

## Results

Every number is from a CPU run on AMD EPYC 7R13, torch 2.15.0. Scripts 02, 04 and 05 used 8 threads. Scripts 01 and 06
and the five per-scheme runs of 03 used 4 threads each, in parallel, on a busy machine, so the training minutes in
`results/03_extend_*.txt` vary by a factor of three between runs and mean nothing. Each file in `results/` starts with
its hardware and command. `results/03_extend.txt` is the merge of the five per-scheme files; its hardware line says
8 threads because the merge itself ran that way.

Tests: 31 passed (offline, about 20 s). There are no `network`, `slow` or `gpu` tests.

### The checks that hold exactly

| Check | Result |
|---|---|
| One attention layer, `max abs(f(Px) - P f(x))`: no mask, no position / causal mask / RoPE | 6e-8 / 0.75 / 0.085 |
| A uniform causal head: component on the first token at position 63 | 0.01562, equal to 1/(i+1) at every position tried |
| Sinusoidal: `max abs(PE(p+k) - R_k PE(p))`, k = 1, 7, 100 | about 1e-13 |
| RoPE: score(m+s, n+s) minus score(m, n), norm change at position 12345 | 1.9e-3 (float32 angle rounding at s = 5000), 1e-6 |
| Interleaved and half-split RoPE related by one permutation | difference 0 |
| Wrong convention, relative score error at distance 1 / 8 / 64 | 0.33 / 0.92 / 1.24 |
| RoPE angle error at position 500,000, base 500,000: median plane, float32 / bfloat16 | 2e-5 / 0.42 rad |
| 128-dim head, base 10,000: planes that never finish one turn inside 4k / 32k / 128k | 18 / 4 / 0 of 64 |
| YaRN frequencies and logit scale vs `transformers` | equal (tests, two configurations) |
| M-RoPE ids vs `Qwen2VLModel.get_rope_index`, two layouts | identical |
| Text-only M-RoPE vs 1D RoPE | difference 0 |
| Interleaved layout vs `transformers` Qwen3-VL frequency recomposition | identical (test) |
| 2D RoPE score for the same offset at three places in the grid | -6.16133 at all three |
| Fourier kernel vs `exp(-2 pi^2 sigma^2 d^2)`, sigma 1, 4096 features, d = 0.1 / 0.3 | 0.8218 vs 0.8209 / 0.1794 vs 0.1692 |

### Trained models (script 03, toy scale)

Two-layer decoder, d = 128, 4 heads, trained 2000 steps at length 64 on an induction-copy task (a 12-token segment
is copied later in the sequence; accuracy is on the copied tokens). Evaluated at 64, 128 and 256 tokens with no
fine-tuning. Mean of two seeds; both seeds are in `results/03_extend.txt`.

| Scheme | acc at 64 | acc at 128 | acc at 256 |
|---|---|---|---|
| RoPE | 0.987 | 0.778 | 0.297 |
| RoPE + Position Interpolation | 0.987 | 0.056 | 0.027 |
| RoPE + NTK-aware base | 0.987 | 0.904 | 0.537 |
| RoPE + YaRN | 0.987 | 0.938 | 0.829 |
| ALiBi | 0.923 | 0.877 | 0.822 |
| Learned table | 0.906 | no row exists | no row exists |
| Sinusoidal | 0.046 | 0.017 | 0.013 |
| NoPE | 0.054 | 0.033 | 0.023 |

Read the table with care.

- RoPE falls off past its training length, and ALiBi barely does. NTK-aware scaling and YaRN recover part or most
  of the loss without any training.
- Position Interpolation collapses here. The post says PI needs a few hundred steps of fine-tuning, and this lab did
  not fine-tune. The toy task needs an exact one-token offset, which PI blurs by crowding adjacent positions. This
  does not test the claim that fine-tuning fixes it.
- Sinusoidal and NoPE did not learn the copy task in 2000 steps (training loss about 4.1 to 4.6, against 0.05 for
  RoPE). So their rows say nothing about length generalisation, and this lab does not reproduce the claim that NoPE
  generalises better than ALiBi, T5-bias and RoPE.
- Two seeds is a small sample. RoPE at 256 tokens was 0.196 and 0.397 across the two seeds.

### Position from the causal mask (script 01)

A 2-layer model with no positional encoding, shown random tokens and asked for each token's own index (0 to 63,
chance 0.016), 1500 steps.

| Model | exact accuracy | mean absolute error, in positions |
|---|---|---|
| NoPE, causal mask | 0.235 | 1.85 |
| NoPE, bidirectional | 0.013 | 21.8 |
| Learned table, causal | 1.0 | 0 |

The causal model recovers position to within about two tokens. The bidirectional model cannot, as the symmetry
argument predicts.

### Fourier features (script 06)

A coordinate MLP fits a synthetic 128x128 image, trained on every other pixel, scored on the rest (2000 steps).
PSNR is capped at 120 dB by the loss floor.

| Input | train PSNR | held-out PSNR |
|---|---|---|
| Raw (x, y) | 19.7 | 19.4 |
| Fourier, sigma 1 | 40.3 | 24.4 |
| Fourier, sigma 10 | 120 | 24.5 |
| Fourier, sigma 100 | 120 | 11.9 |

Too small a sigma blurs, a very large one memorises. A click at (0.413, 0.772) matches the nearest grid cell of a
64x64 grid under one shared random B. Two clicks in the same cell have different encodings (cosine 0.9991, not 1).

### M-RoPE (script 05)

| Measurement | Result |
|---|---|
| Wavelength range of the temporal axis, 128-dim head, base 10^6: chunked [16, 24, 24] / interleaved [24, 20, 20] | 6.3 to 160 / 6.3 to 5.1e6 |
| Temporal id of second 6 in a clip at 1 fps / 4 fps: frame index / absolute time | 3 and 12 / 12 and 12 |
| Position range used by a 2048x2048 image: flat ids / M-RoPE / V2PE step 1/16 | 5476 / 74 / 342 |
| Text after an image: `max + 1` rule / `previous id + 1` rule | starts at 8 / starts at 5, reusing 3 image ids |

The `previous id + 1` row uses the temporal axis, which is nearly flat inside an image. The V2PE number is
arithmetic on the 1/16 step the post quotes, not a trained model.

## What was not measured

- No trained model at the scale of the papers. Part B is a 2-layer model on a copy task. The results for YaRN,
  PI and NTK in the post come from billion-parameter models that were also fine-tuned, and none of that was run.
- Fine-tuning after extension, LongRoPE, PoSE, SelfExtend, p-RoPE, partial rotary, FIRE, KERPLE, T5 buckets and
  CoPE were not built.
- Lost in the middle, attention sinks and cache eviction were not reproduced.
- No pretrained vision tower or VLM was loaded. The resize check uses a synthetic table, so it shows the cls-row
  mistake and the antialias gap, not DINOv2's reported numbers.
- No GPU was used. Nothing here is a speed measurement.
