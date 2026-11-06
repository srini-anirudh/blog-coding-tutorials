# Lab: Inside a Modern Vision Encoder

The code companion to the theory post [Inside a Modern Vision Encoder](https://srini-anirudh.github.io/blogs/inside-a-modern-vision-encoder/).
The post argues that what changed in vision encoders is how pixels get sampled, sliced, supervised and counted.
This lab builds those pieces in small Python and runs them, so you can watch each claim happen.

What you can run:

1. **The intake bench.** Seven intake strategies (CLIP crop, squash, letterbox, LLaVA-NeXT AnyRes, InternVL tiles,
   Qwen2-VL native resolution, SigLIP 2 NaFlex) rebuilt from their processors and checked against the reference code
   in `transformers`. For any image size you get the token count, the stretch, the crop, the padding and how many bars
   of a resolution target survive.
2. **Preprocessing gotchas.** Resizing without antialiasing, PIL against torch, the wrong normalisation constants,
   EXIF rotation and alpha flattening, each measured on synthetic images.
3. **Pixel shuffle.** 1024 patches become 256 tokens with nothing lost, against 2 x 2 average pooling, which loses detail.
4. **CLIP softmax against SigLIP sigmoid.** Which gradients a single cell touches, the block-by-block SigLIP loss, and a
   toy batch-size sweep.
5. **Patch n' Pack.** A tiny 2D-RoPE ViT that packs images of different shapes into one sequence, what a missing mask
   does, and how much padding packing removes.
6. **Registers on real checkpoints.** High-norm outlier tokens in DINOv2 small, base and large, with and without
   4 register tokens.
7. **The preprocessing tax on a real encoder.** DINOv2 kNN accuracy on 1,000 mini-ImageNet images, with one
   preprocessing step broken at a time.

## Layout

```
labs/inside-a-modern-vision-encoder/
├── vencoder/
│   ├── intake.py       # the seven strategies, the resolution target, bar measurement, ViT FLOPs
│   ├── preprocess.py   # resamplers, normalisation constants, EXIF, alpha
│   ├── shuffle.py      # pixel shuffle (imported from C7) and its inverse, average pooling
│   ├── contrastive.py  # CLIP and SigLIP losses, chunked SigLIP, gradient coupling, toy two-tower training
│   ├── packing.py      # tiny ViT with 2D RoPE, packing, padding, masks, first-fit cost
│   ├── realdata.py     # mini-ImageNet slice, DINOv2 loading, token norms, kNN
│   └── common.py       # hardware line and results writer
├── scripts/            # 01 to 07, one per mechanism, plus run_all.sh
├── tests/
└── results/            # real outputs; every file starts with the hardware and the command
```

Pixel shuffle is not rebuilt here. It comes from C7, [`c07-vlm/tinyvlm/projector.py`](../../c07-vlm/tinyvlm/projector.py),
whose tests check it against Idefics3. C7 also covers the full image processor, the projector and the multimodal
sequence, so this lab skips them.

## Setup

From the repository root, with the repo's environment (`pip install -e ".[dev]"`). No extra packages. Scripts 01 to 05
need no downloads. Scripts 06 and 07 download six DINOv2 checkpoints (about 3.2 GB) and read 1,000 mini-ImageNet
validation images (about 100 MB) from the Hugging Face Hub. They cache the images in `data_cache/`, which git ignores.

## Run everything

```bash
bash labs/inside-a-modern-vision-encoder/scripts/run_all.sh            # all seven scripts
bash labs/inside-a-modern-vision-encoder/scripts/run_all.sh offline    # 01-05 only, no downloads

python labs/inside-a-modern-vision-encoder/scripts/01_intake_bench.py
python labs/inside-a-modern-vision-encoder/scripts/02_preprocessing.py
python labs/inside-a-modern-vision-encoder/scripts/03_pixel_shuffle.py
python labs/inside-a-modern-vision-encoder/scripts/04_contrastive.py      # --samples, --seeds, --batches
python labs/inside-a-modern-vision-encoder/scripts/05_patch_n_pack.py
python labs/inside-a-modern-vision-encoder/scripts/06_registers.py        # network; --images, --sizes
python labs/inside-a-modern-vision-encoder/scripts/07_preprocessing_tax.py  # network; --models, --k

# Tests
python -m pytest labs/inside-a-modern-vision-encoder -m "not slow and not gpu and not network"
python -m pytest labs/inside-a-modern-vision-encoder -m network
```

## Which script covers which section of the post

| Post section | Script |
|---|---|
| The intake bench | `01_intake_bench.py` (bench table) |
| The six knobs: patch size | `01_intake_bench.py` (p = 14 against p = 16) |
| Preprocessing: resize, aspect handling, normalise, decode and orient | `02_preprocessing.py`, `07_preprocessing_tax.py` |
| Preprocessing: snap to a patch multiple, set the token budget | `01_intake_bench.py` (Qwen and NaFlex rows, the `max_pixels` cap) |
| Preprocessing: pad to a packed sequence, build the mask | `05_patch_n_pack.py` |
| Five ways to feed a document, and the token cost calculator | `01_intake_bench.py` |
| Pixel shuffle | `03_pixel_shuffle.py` |
| Contrastive: softmax coupling against sigmoid independence | `04_contrastive.py` |
| Encoder walkthroughs: "divergence happens early" (196 against 9,804 patches) | `01_intake_bench.py` (walkthrough page) |
| Field notes: your ViT has about 2% garbage tokens | `06_registers.py` |
| Field notes: "any resolution" has an envelope (position) | `05_patch_n_pack.py` (2D RoPE), `vencoder/packing.py` (`interpolate_pos_embed`) |
| Qwen2.5-VL window attention | `01_intake_bench.py` (ViT FLOPs) |

## Results

Every number below was measured on an **AMD EPYC 7R13 CPU**, torch 2.15.0, float32. Scripts 01 to 03 and 05 to 07
used 8 threads. Script 04 used 2 threads. The machines were shared with other jobs, so the few timings here are rough.
Accuracies, counts and norms do not depend on load. Raw outputs are in `results/`, each file headed by its hardware and
command. No GPU was used: nothing in this post is about speed.

### 1. The intake bench (`01_intake_bench.txt`)

A synthetic page, 1024 px on its long side, carries five bar groups (3 dark bars each, both orientations) at strokes of
24, 15, 9, 5 and 3 px, as on the post's bench. "Resolved" is our measurement: a group counts when at least half of the
black-to-white swing survives in both orientations, at the worst of four pixel phases. "Proxy" is the post's rule, a
stroke of at least 1.6 output pixels. Groups that were cropped away count as lost in both.

| Image | Strategy | Patches | LLM tokens | Stretch | Cropped | Padding | Resolved | Proxy |
|---|---|---|---|---|---|---|---|---|
| A4 page 1:1.41 | CLIP crop 224 | 256 | 256 | 0% | 29% | 0% | 3/5 | 2/5 |
| A4 page 1:1.41 | Squash 336 | 576 | 576 | 41% | 0% | 0% | 4/5 | 4/5 |
| A4 page 1:1.41 | Letterbox 384 | 729 | 729 | 0% | 0% | 29% | 5/5 | 4/5 |
| A4 page 1:1.41 | AnyRes tiles | 2,880 | 2,880 | 0% | 0% | 29% | 5/5 | 5/5 |
| A4 page 1:1.41 | InternVL tiles | 13,312 | 3,328 | 6% | 0% | 0% | 5/5 | 5/5 |
| A4 page 1:1.41 | Native (Qwen) | 3,848 | 962 | 1% | 0% | 0% | 5/5 | 5/5 |
| A4 page 1:1.41 | NaFlex 1024 | 999 | 999 | 3% | 0% | 0% | 5/5 | 5/5 |
| Receipt 1:3.2 | CLIP crop 224 | 256 | 256 | 0% | 69% | 0% | 1/5 | 1/5 |
| Receipt 1:3.2 | Squash 336 | 576 | 576 | 220% | 0% | 0% | 4/5 | 4/5 |
| Receipt 1:3.2 | Letterbox 384 | 729 | 729 | 0% | 0% | 69% | 5/5 | 4/5 |
| Receipt 1:3.2 | Native (Qwen) | 1,628 | 407 | 5% | 0% | 0% | 5/5 | 5/5 |

The full table has all five shapes and seven strategies. Three things to notice. Centre crop loses bar groups because it
throws the ends of the page away, not because it samples too coarsely. Squash keeps everything but stretches a receipt
by 220%, and the 3 px bars drop to 8% of their swing on the squeezed axis. At a 1024 px master, every strategy that
keeps the whole page at 0.375 scale or above resolves the 3 px bars by our measure. The post's 1.6 px proxy is stricter
than our swing measure: letterbox keeps 81% of the swing at 1.1 output pixels.

Token counts match the post: CLIP ViT-L/14 at 336 gives 576; an InternVL tile is 1,024 patches and 256 tokens.
On the walkthrough page (1200 x 1600), DINO v1 at 224 sees **196 patches** and Qwen2-VL sees **9,804** (2,451 after the
2 x 2 merge), the post's fiftyfold spread. That count uses `max_pixels = 12,845,056` from the Qwen2-VL and Qwen2.5-VL
`preprocessor_config.json` on the Hub. The `transformers` code default is `max_pixels = 1,003,520`. With it, the same
page gets 4,920 patches, half as many. This is the post's "serving caps" warning, in one config value.

Token cost calculator, patches -> LLM tokens (the receipt size, 600 x 3000, is our choice):

| Image | CLIP crop | Squash | Letterbox | AnyRes | InternVL | Qwen | NaFlex |
|---|---|---|---|---|---|---|---|
| 1 MP square | 256 | 576 | 729 | 2,880 | 10,240 -> 2,560 | 5,476 -> 1,369 | 1,024 |
| A4 @ 200 dpi | 256 | 576 | 729 | 2,880 | 7,168 -> 1,792 | 19,824 -> 4,956 | 999 |
| 1440p screenshot | 256 | 576 | 729 | 2,880 | 9,216 -> 2,304 | 18,564 -> 4,641 | 1,008 |
| 4K frame | 256 | 576 | 729 | 2,880 | 9,216 -> 2,304 | 42,196 -> 10,549 | 1,008 |

A 4K frame is 42,196 Qwen patches, above the post's "more than 20,000 tokens before compression". With the Qwen2.5-VL
vision tower's real config (32 blocks, width 1280, 8 x 8-patch windows, full attention in 4 blocks), window attention
cuts the ViT's forward FLOPs by 1.95x on the walkthrough page and by 3.83x on the 4K frame. These are FLOP counts, not
timings. Patch size 14 gives 30.6% more tokens than patch size 16 at equal pixels.

The AnyRes counts are before LLaVA-NeXT removes padded features and adds row-end tokens.

### 2. Preprocessing (`02_preprocessing.txt`)

| Measurement | Result |
|---|---|
| 4000 px page of 3 px strokes shrunk to 336. Correct answer: flat grey. Std of output: PIL bicubic / torch antialias=True / torch antialias=False / PIL nearest | 0.003 / 0.002 / **0.444** / **0.499** |
| Same, RMSE against an area average | 0.015 / 0.014 / 0.450 / 0.505 |
| PIL bicubic against torch bicubic, 1024x768 -> 224x168, max difference in 8-bit levels: torch antialias=True / False | 3 / 146 |
| ImageNet constants fed to a SigLIP model: gain and offset per channel (R, G, B) | gain 2.18, 2.23, 2.22; offset +0.07, +0.20, +0.42 |
| EXIF orientation 6: size from `Image.open` / after `ImageOps.exif_transpose` | 64x32 / 32x64 |
| Black lines on a transparent PNG: line contrast after `convert('RGB')` / composite on white | 0.00 / 1.00 |

Without antialiasing, a page of small type does not become grey. It becomes stripes that are not in the page. Torch's
antialiased bicubic agrees with PIL to within 3 levels on this image; the non-antialiased one is off by up to 146.
`convert('RGB')` on a transparent chart keeps the hidden black RGB under the transparent pixels, so the lines vanish.

### 3. Pixel shuffle (`03_pixel_shuffle.txt`)

| Input | Patches x dim | Tokens x dim | Max error after the inverse |
|---|---|---|---|
| InternVL tile, r = 2 | 1024 x 1024 | 256 x 4096 | 0 |
| SmolVLM-256M crop, r = 4 (from its `config.json`) | 1024 x 768 | 64 x 12288 | 0 |
| 27 x 27 grid, r = 3 | 729 x 1152 | 81 x 10368 | 0 |

On a 448 px tile of bars, cut into raw 14 px patches as the "features", pixel shuffle and its inverse give back the
tile exactly (RMSE 0, every bar group at full swing). Averaging each 2 x 2 block of tokens instead gives RMSE 0.239,
and four of the five bar groups keep only half of their swing. The post names r = 3 for SmolVLM. The SmolVLM-256M
checkpoint in our cache uses r = 4.

### 4. CLIP softmax against SigLIP sigmoid (`04_contrastive.txt`)

Raise one off-diagonal logit of an 8 x 8 batch. Under the softmax loss, 14 other cells' gradients change: the rest of
that row and that column. Under the sigmoid loss, none do. The SigLIP loss computed in 2, 4 or 8 blocks per side equals
the full-matrix loss to within 2e-6, while holding at most 1/64 of the logits at once. That is why SigLIP needs no
all-gather of the full similarity matrix.

Toy two-tower retrieval. Two noisy non-linear views of one 16-dimensional latent, MLP towers, 200,000 pairs seen per
run whatever the batch size, top-1 among 2,000 held-out pairs (chance 0.0005), 3 seeds:

| Batch | Steps | CLIP softmax | SigLIP sigmoid |
|---|---|---|---|
| 8 | 25,000 | 0.381 +/- 0.012 | 0.437 +/- 0.009 |
| 32 | 6,250 | 0.441 +/- 0.007 | 0.460 +/- 0.004 |
| 128 | 1,562 | 0.466 +/- 0.001 | 0.477 +/- 0.002 |
| 512 | 390 | 0.486 +/- 0.001 | 0.487 +/- 0.002 |

The sigmoid loss is ahead at small batches and the gap closes by batch 512. That is the shape of the SigLIP paper's
batch-size result, at toy scale. The paper's batches of 32k and beyond, and its finding that accuracy stops improving
past about 32k, were not reproduced.

### 5. Patch n' Pack (`05_patch_n_pack.txt`)

A tiny ViT (3 blocks, width 64, patch 8, 2D RoPE, random weights). Four images of different shapes packed into one
108-patch sequence with a block-diagonal mask give the same features as encoding each alone (max difference 1e-6).
Without the mask, features move by up to 0.32. A 16-patch image in a padded batch:

| Row length | Padding | Pooled cosine, masked | Pooled cosine, no mask |
|---|---|---|---|
| 32 | 50% | 1.0000 | 0.8064 |
| 64 | 75% | 1.0000 | 0.5843 |
| 256 | 94% | 1.0000 | 0.4007 |

Nothing crashes without the mask. The pooled feature just drifts further from the right one as padding grows.
On 512 images at native resolution (long side 400 to 1600 px, six aspect ratios, Qwen2-VL patch counts), padding to
the longest image in batches of 8 wastes 56% of the slots. First-fit packing wastes 1%. A block-diagonal attention
kernel computes 3.5x fewer query-key pairs than dense padded attention. 2D RoPE gives the same q.k score for the same
offset at three different absolute positions, which is what lets it handle any grid shape.

### 6. Registers on real DINOv2 checkpoints (`06_registers.txt`, network)

160 mini-ImageNet validation images at 224 px, 256 patches each. Norms are of each block's output, before the final
LayerNorm. An outlier is a patch token with more than 3x its image's median norm. Each model is read at the block where
its no-register sibling's outliers stand out most.

| Model | Block | Outlier tokens | Images with an outlier | Largest token / its 8 neighbours | Outliers, last block |
|---|---|---|---|---|---|
| dinov2-small | 11 of 12 | 0.21% | 41% | 2.3x | 0% |
| dinov2-with-registers-small | 11 of 12 | 0% | 0% | 1.4x | 0% |
| dinov2-base | 9 of 12 | **1.66%** | **100%** | **53x** | 0.25% |
| dinov2-with-registers-base | 9 of 12 | 0% | 0% | 1.2x | 0% |
| dinov2-large | 20 of 24 | **0.40%** | **100%** | **28x** | 0% |
| dinov2-with-registers-large | 20 of 24 | 0% | 0% | 1.2x | 0% |

The artifacts are real and the registers remove them. In base, about 4 tokens per image (1.66%) carry 53 times the norm
of their neighbours. In large, exactly one token per image does, through blocks 18 to 23. Small barely has them, which
fits the registers paper's finding that the artifacts come with larger models. Two qualifications to the post's
"about 2% of tokens at about 10x". The fraction and the ratio vary by model size: we see 0.4% to 1.7%, at 28x to 53x.
And in these checkpoints the outliers are in the late blocks, not the final one: at the last block's output they are
mostly gone. If you look only at the final layer of DINOv2-large, you will not find them.

### 7. The preprocessing tax on a real encoder (`07_preprocessing_tax.txt`, network)

1,000 mini-ImageNet validation images (20 classes, 50 each, median long side 500 px), leave-one-out 10-nearest-neighbour top-1 on
the CLS feature. One preprocessing step is changed per row. "Cosine" is the mean cosine between each image's CLS feature
and its CLS feature under the correct pipeline.

| Preprocessing | dinov2-small kNN | Cosine | dinov2-base kNN | Cosine |
|---|---|---|---|---|
| Correct: bicubic with antialiasing, resize 256, crop 224, ImageNet constants | 0.965 | 1.000 | 0.989 | 1.000 |
| SigLIP constants (0.5 / 0.5) | 0.967 | 0.984 | 0.988 | 0.989 |
| OpenAI CLIP constants | 0.964 | 0.999 | 0.988 | 0.999 |
| torch bicubic, antialias=False | 0.967 | 0.995 | 0.986 | 0.996 |
| PIL nearest | 0.966 | 0.975 | 0.987 | 0.979 |
| Squash to 224 instead of crop | 0.967 | 0.930 | 0.986 | 0.940 |

Here the tax is small. Every broken step moves the features (cosine down to 0.93), but kNN accuracy moves by at most
0.3 points, inside the noise of 1,000 images. The post's claim that a wrong constant can cost more than a 10x larger
model is not supported by this test. Three reasons it may not transfer. The task is easy: 20 ImageNet classes, and
the scores sit near the ceiling. The images are small (median long side 500 px), so shrinking them to 256 aliases little; the
synthetic 4000 px page in section 2 is where antialiasing bites. And kNN on a global feature is a forgiving probe;
the post's examples are documents and dense tasks inside a VLM, which this lab does not run.

## What was not measured

- No VLM benchmark numbers. The post's DocVQA, OCRBench, MMVP and ImageNet zero-shot figures, and every claim about
  SigLIP 2 against InternViT-6B, are paper results at scales this lab does not reach.
- No real contrastive pretraining. Section 4 is a toy two-tower problem; it shows the mechanism and the trend, not CLIP
  or SigLIP accuracy.
- No real Qwen, InternVL, LLaVA-NeXT or SigLIP 2 model was run. Their intake geometry is rebuilt and checked against the
  `transformers` reference functions; their encoders were not executed.
- Window attention is counted in FLOPs, not timed.
- Penultimate-layer tapping, DINOv2 versus CLIP features (CLIP-blind pairs), position-embedding interpolation far from
  the training grid, and video frame sampling are explained in the post but not measured here. `interpolate_pos_embed`
  is provided and tested, with no accuracy experiment behind it.

## Tests

```bash
python -m pytest labs/inside-a-modern-vision-encoder -m "not slow and not gpu and not network"   # 57 tests, about 20 s
python -m pytest labs/inside-a-modern-vision-encoder -m network                                  # 2 tests
```

The offline tests check `smart_resize`, NaFlex sizing, the InternVL grid choice and the AnyRes grid choice against the
functions in the installed `transformers` (the Qwen and SigLIP 2 files are read with `ast`, because their modules need
torchvision). They also check the post's token counts, the pixel shuffle round trip, the CLIP loss against a manual
version, chunked SigLIP against full SigLIP, packed against separate encoding, pad masks, 2D RoPE's relative property,
and the preprocessing effects. The network tests load DINOv2 with registers and check the Qwen2.5-VL constants used in
section 1 against its Hub config. There are no `slow` or `gpu` tests.

## Theory

- [Inside a Modern Vision Encoder](https://srini-anirudh.github.io/blogs/inside-a-modern-vision-encoder/)
- C7, [Build a Vision-Language Model From Scratch](../../c07-vlm/), for the processor, projector and multimodal sequence.
