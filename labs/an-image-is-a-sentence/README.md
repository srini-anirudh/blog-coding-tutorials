# Lab: An Image Is a Sentence

Code companion for the theory post
[An Image Is a Sentence: How Vision Transformers See](https://srini-anirudh.github.io/blogs/an-image-is-a-sentence/).

The post explains the vision transformer family through its mechanisms: patchify, global attention and its
quadratic cost, Swin's shifted windows, PVT's key downsampling, MViT's pooling, ConViT's gate, DeiT's
distillation token and the space-time schemes for video. This lab builds each one as a small PyTorch module and
runs it. It checks the post's cost formulas against FLOPs counted from real modules. It computes exactly how far
each block type can reach over depth. It trains small models on CIFAR-10 to show the small-data side of the
inductive-bias argument. It also measures a pretrained DeiT.

Swin's window partition, cyclic-shift mask and window attention are imported from the
[C3 attention tutorial](../../c03-attention/) (`attnzoo.vision`). This lab does not rebuild them.

## Layout

```
labs/an-image-is-a-sentence/
├── vitfamily/
│   ├── patch.py       # PatchEmbed, patchify as one matmul, patch merging, tubelets, position-table resize
│   ├── attention.py   # MHSA, PVT spatial-reduction attention, MViT pooling attention, ConViT GPSA
│   ├── blocks.py      # pre-norm ViT block, PVT block, Swin block (C3's WindowAttention), divided space-time block
│   ├── models.py      # TinyViT (CLS, GAP, distillation token, GPSA layers) and TinyCNN
│   ├── flops.py       # the post's formulas, and a FlopCounterMode measurement split into linear / attention / conv
│   ├── reach.py       # boolean reach over depth: global, windows, shifted windows, axial, conv, space-time schemes
│   ├── stats.py       # mean attention distance
│   ├── data.py        # CIFAR-10 and tiny-imagenet from the Hugging Face Hub
│   ├── report.py      # writes results/ with a hardware line and the command first
│   └── c03.py         # imports from the C3 tutorial
├── scripts/           # 01 to 06, listed below
├── results/           # real outputs of every script
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Scripts 05 and 06 download from the Hugging Face Hub
once: CIFAR-10 (140 MB), the tiny-imagenet validation split (15 MB) and `facebook/deit-tiny-distilled-patch16-224`
(24 MB). Decoded images are cached as tensors under `$HF_HOME/vitfamily/`.

## Run everything

```bash
# Tests.
python -m pytest labs/an-image-is-a-sentence -m "not slow and not gpu and not network"   # offline, a few seconds
python -m pytest labs/an-image-is-a-sentence -m "slow or network"                        # downloads; under a minute

# Scripts. Each prints its results and writes results/<script name>.txt (and .json).
python labs/an-image-is-a-sentence/scripts/01_tokens.py          # seconds
python labs/an-image-is-a-sentence/scripts/02_flop_ledger.py     # seconds, on the meta device
python labs/an-image-is-a-sentence/scripts/03_reach.py           # seconds
python labs/an-image-is-a-sentence/scripts/04_wallclock.py --device cpu
python labs/an-image-is-a-sentence/scripts/04_wallclock.py --device cuda
python labs/an-image-is-a-sentence/scripts/05_small_data.py --device cuda   # 12 runs, 18 min on one H100
python labs/an-image-is-a-sentence/scripts/05_small_data.py --quick         # a short CPU smoke run
python labs/an-image-is-a-sentence/scripts/06_pretrained_deit.py            # CPU, about 18 min on a shared EPYC
```

## Which script covers which section of the post

| Post section | Script |
|---|---|
| Patchify: the only new layer | `01_tokens.py` (conv vs matmul, CvT's overlapping embed) |
| Anatomy of a ViT block | `01_tokens.py` (MLP holds two thirds of the weights) |
| Inside one attention head | `01_tokens.py` (why divide by sqrt(d)) |
| Position: four answers | `01_tokens.py` (permutation equivariance), `06_pretrained_deit.py` (224 to 384 resize, and the bug) |
| The quadratic wall | `02_flop_ledger.py`, `04_wallclock.py` part 2 |
| The missing inductive bias | `05_small_data.py`, `06_pretrained_deit.py` (mean attention distance) |
| DeiT and the data problem | `05_small_data.py`, `06_pretrained_deit.py` |
| PVT: the pyramid returns | `02_flop_ledger.py` |
| Swin: windows, and the shift; patch merging | `03_reach.py`, `01_tokens.py` |
| Deriving Swin's complexity | `02_flop_ledger.py` |
| CvT, ConViT, CoAtNet | `01_tokens.py` (GPSA at init), `05_small_data.py` (ConViT trained) |
| MViT: pooling attention | `02_flop_ledger.py` |
| Video: tubelets | `01_tokens.py` |
| TimeSformer's five schemes | `03_reach.py`, `02_flop_ledger.py` |
| ViViT's four factorisations | `03_reach.py` (zero-init temporal projection) |
| Lab: the complexity bench | `02_flop_ledger.py` |
| Lab: reach and receptive field | `03_reach.py` |
| Training cost vs inference cost | `04_wallclock.py`, `02_flop_ledger.py` (activation memory) |

## Results

Every number names its hardware. Raw output is in `results/`, under the name of the script that wrote it. Each
file starts with the hardware line and the command.

Tests on AMD EPYC 7R13, 8 threads, torch 2.15.0: 27 offline tests pass in about 3 s. The 3 `slow` and `network`
tests also pass (43 s). There are no `gpu` tests; the CUDA path is the `--device cuda` scripts.

### Exact checks (no hardware dependence)

These are counts and identities, not timings. FLOPs are counted by `torch.utils.flop_counter.FlopCounterMode` on
modules built on the `meta` device. They count matmuls and convolutions only, as the post does.

| Check (post section) | Result | Post |
|---|---|---|
| Patchify conv vs reshape + matmul, 224 px, P=16, D=768 | max difference 3.1e-6 (float32 rounding) | same operation |
| ViT block FLOPs vs 24ND^2 + 4N^2D, N = 197 to 4625 | exact match at every N | formula |
| Attention share of ViT-B block FLOPs at 224 / 384 / 1024 px | 0.041 / 0.111 / 0.471 | crossover at N = 6D |
| Crossover N = 6D at D=768 | 4608 tokens = 1086 px at patch 16 | "roughly 1088" |
| MLP share of attention + MLP weights, ViT-B block | 0.667 | two thirds |
| Swin stage 1 attention sublayer, MACs: global / window | 2003.8 M / 145.1 M, ratio 13.8x | 2 003 / 145, 13.8x |
| Swin stage 1, attention term alone | 1888.2 M vs 29.5 M MACs (64x) | 1 887 M vs 29.5 M |
| Swin stage 1, full block with the MLP | 2235.0 M vs 376.3 M MACs, ratio 5.9x | not stated |
| Swin stage 4 attention share of the attention sublayer | 0.031 | 3% |
| PVT stage 1 global attention, 56x56, D=64 | 2.52 GFLOPs per layer | about 2.5 G |
| PVT SRA block, R=8, everything except attention | 283.4 M FLOPs, matching our op-by-op count | 24ND^2 + ND^2/R^2 = 308.5 M |
| Joint / divided space-time pairs, T=8 / T=32, N=196 | 7.7x / 27.5x | 7.7x / 32x |
| Divided block projections + MLP | 32 TND^2 | 48 TND^2 |
| ViT-L at 1024 px: naive scores / all other activations | 4.9x | "100x larger" |
| ViT-B score matrix at 1024 px, 12 heads, fp16 | 201.4 M elements, 403 MB per layer | 201 M, 400 MB |
| Swin (W-MSA then SW-MSA): blocks for the corner token to reach all of a 56x56 map | 15 | about 16 |
| Shifted-window reach radius per block, L = 1 to 7 / conv 3x3 | 3.5 / 1.0 tokens per block (3.5x) | M/2 per block pair, 3.5x |
| Unshifted windows: share of the map ever reached | 1.6% at every depth | plateau |
| Axial (row, column): blocks to reach everything | 2 | 2 |
| SW-MSA without the mask: corner reaches far corner after 2 blocks | yes (wrap-around); with the mask, no | the torus bug |
| Divided block with zero-init temporal projection vs image block | max difference 0.0 | exact at init |
| Patch merging vs PixelUnshuffle | same numbers up to a fixed channel order | same operation |
| Score variance at d=64: unscaled / divided by sqrt(d) | 63.6 / 0.99 | d / 1 |
| Mean attention distance at init, 224 px: fresh MHSA / GPSA positional term | 116.5 px (= uniform) / 20.2 px | global / convolutional |

### Small-data training: NVIDIA H100 80GB HBM3, torch 2.15.0, bf16

CIFAR-10 test set (10,000 images). 3000 AdamW steps of batch 128 at every training-set size, so the small subsets
see many more epochs. lr 1e-3 with cosine decay, weight decay 0.05, random crop and flip. One seed per cell. The
DeiT teacher is the CNN trained on the same subset, with hard labels. The whole run took 18 minutes on one GPU.

| Training images | TinyCNN (1.08 M params) | TinyViT (1.21 M) | ConViT, 4 GPSA blocks (1.21 M) | DeiT, distilled from the CNN (1.21 M) |
|---|---|---|---|---|
| 1,000 | 0.614 | 0.400 | 0.463 | 0.416 |
| 5,000 | 0.792 | 0.580 | 0.655 | 0.578 |
| 50,000 | 0.870 | 0.699 | 0.758 | 0.712 |

- The CNN wins at every size here. The gap to the plain ViT shrinks from 21 points to 17 as the data grows 50x.
- ConViT's soft convolutional prior recovers part of that gap at every size (+6 to +8 points over the ViT).
- ConViT's gates hardly moved: sigmoid(lambda) ended between 0.69 and 0.74, from 0.731 at init. At this scale and
  budget, no head learned its way out of the positional prior.
- Distilling from the CNN gave at most +1.6 points over the plain ViT. The two DeiT heads disagree on only 2 to 3% of
  test images, and their final embeddings have cosine 0.999. With a teacher this close to the labels, the two tokens
  learned almost the same thing.

### Pretrained DeiT-tiny (distilled): AMD EPYC 7R13, 8 threads, torch 2.15.0, transformers 5.22.2

1000 tiny-imagenet validation images, 64 px, upsampled to 224 or 384. The images are low resolution, so absolute
agreement numbers are low for every variant. Compare the variants with each other.

| Measurement | Result |
|---|---|
| Mean attention distance, layer 1, three heads | 81.0 / 84.9 / 38.7 px |
| Mean attention distance, layer 4, three heads | 31.0 / 21.7 / 56.6 px |
| Mean attention distance, layers 9 to 12, every head | 98 to 119 px (uniform attention: 116.5 px) |
| Cosine of the learned CLS and distillation tokens (input parameters) | 0.452 |
| Cosine of their hidden states after layers 1 / 6 / 12 | 0.94 / 0.97 / 0.90 |
| The two heads' top-1 predictions disagree | 14.0% of images |
| 384 px, our position resize vs transformers' `interpolate_pos_encoding` | identical predictions (100%) |
| 384 px top-1 agreement with the 224 px prediction: correct resize / CLS-row bug | 47.6% / 19.3% |

### Wall clock: one block at Swin's stage-1 shape (56 x 56 tokens, D = 96)

FLOPs come from the counter above. GFLOP/s is measured FLOPs divided by measured time. The Swin rows use C3's
teaching implementation of window attention. It rebuilds the shift mask on the CPU on every call and passes a float
bias to SDPA, which rules out the fused flash kernel. So these rows measure that implementation, not a tuned Swin
kernel. The ViT row at 196 tokens and D=384 has exactly the same FLOPs as the Swin block (0.753 G), because
3136 x 96^2 = 196 x 384^2.

| Block | GFLOPs / sample | H100, bf16, batch 64: ms | H100 GFLOP/s | EPYC, fp32, batch 4: ms | EPYC GFLOP/s |
|---|---|---|---|---|---|
| ViT global, 3136 tokens | 4.470 | 2.20 | 130,259 | 247.75 | 72 |
| Swin W-MSA, M=7 | 0.753 | 11.40 | 4,227 | 30.47 | 99 |
| Swin SW-MSA, M=7, shift 3 | 0.753 | 40.53 | 1,188 | 47.40 | 64 |
| PVT SRA, R=8 | 0.697 | 1.43 | 31,276 | 24.01 | 116 |
| ViT global, 196 tokens, D=384 | 0.753 | 0.62 | 77,571 | 15.44 | 195 |

The EPYC was shared with other jobs, so treat its timings as orders of magnitude. On the H100 the global block at
3136 tokens does 6x the FLOPs of the windowed block and still finishes 5x sooner. At equal FLOPs, the plain ViT
block is 18x faster than this window-attention implementation.

One ViT-B attention layer (12 heads, D=768), forward and backward, batch 1, bf16, NVIDIA H100 80GB HBM3. Memory is
the peak allocated above the starting point.

| Image | Tokens | Naive (scores materialised) | Fused SDPA | Naive ms | SDPA ms | One fp16 score matrix |
|---|---|---|---|---|---|---|
| 224 | 197 | 37 MiB | 37 MiB | 2.52 | 1.22 | 1 MiB |
| 512 | 1025 | 104 MiB | 44 MiB | 1.43 | 0.90 | 24 MiB |
| 768 | 2305 | 504 MiB | 62 MiB | 1.96 | 1.53 | 122 MiB |
| 1024 | 4097 | 1567 MiB | 107 MiB | 6.14 | 5.59 | 384 MiB |

### Which machine wrote each file

| Machine | Files in `results/` |
|---|---|
| NVIDIA H100 80GB HBM3, torch 2.15.0 | `04_wallclock_cuda.*`, `05_small_data.*` |
| AMD EPYC 7R13, 8 threads | every other file |

## Where the measurements and the post disagree

These are checks of the post's arithmetic and quoted claims. The post was not changed.

- The Swin table's 13.8x is the attention sublayer alone, in MACs. With the MLP included, the stage-1 block ratio is
  5.9x. The attention term is 1888.2 M MACs; the post writes 1 887.
- The post's SRA row, 24ND^2 + ND^2/R^2, does not match the module. Counted op by op, Q and O cost 4ND^2, K and V
  cost 4ND^2/R^2 on the reduced tokens, the R x R strided conv costs 2ND^2, and the MLP 16ND^2: 22ND^2 + 4ND^2/R^2.
  The counter agrees with the op-by-op count.
- Divided space-time attention adds one attention sublayer, not a second MLP: 32 TND^2, not 48. TimeSformer's own
  block adds a temporal linear layer, which would make it 34.
- At 32 frames the joint/divided pair ratio is TN/(T+N) = 27.5x, not 32x.
- For ViT-L at 1024 px, the post's own formulas give naive scores 4.9x the other activations, not 100x.
- Shifted windows grow the reach radius by about M/2 = 3.5 tokens per block, not per block pair. That rate is what
  makes the post's "about 16 blocks" and "3.5x faster than a 3x3 conv" come out right.
- The post says the DeiT class and distillation tokens' final embeddings have cosine similarity around 0.06. In the
  DeiT paper, 0.06 is the similarity of the learned input tokens; their outputs become similar through the network.
  In DeiT-tiny we measure 0.45 for the input tokens and 0.90 after the last layer.

## What was not measured

- The post's data threshold (ViT overtakes CNNs somewhere around 10 to 30 M images) was not reproduced. CIFAR-10 has
  50,000 training images; the training script shows only the small-data side, with one seed per cell.
- No ImageNet accuracy for any model, and no paper-scale training. DeiT's distillation result (a RegNet teacher
  beating a transformer teacher) was not reproduced; the lab's teacher is a small CNN.
- CvT and CoAtNet blocks, MViT's full block with residual pooling of the skip path, ViViT models 1, 2 and 4, and
  Hiera were not built. MViT and PVT appear as attention layers and in the FLOP ledger only.
- The interactive figures of the post ("one image, eight backbones", the compare rig, the family timeline) were not
  rebuilt.
- Wall-clock numbers cover one block, one shape and one implementation each. No tuned Swin or FlashAttention-3
  kernels, and no end-to-end model latency.
