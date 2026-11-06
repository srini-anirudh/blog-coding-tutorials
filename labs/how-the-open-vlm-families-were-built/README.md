# Lab: How the Open VLM Families Were Built

Code companion for the survey post
[How the Open VLM Families Were Built](https://srini-anirudh.github.io/blogs/how-the-open-vlm-families-were-built/).

The post compares LLaVA, Qwen-VL, InternVL, Molmo, Idefics, Gemma, DeepSeek, MiniCPM-V and their peers along one
checklist. This lab does not train those models. It opens the real ones on the Hugging Face Hub instead. It reads
their processors, their configs and the headers of their weight files, and it checks the post's model-side claims
against what is actually shipped.

Each idea is written from scratch in small Python and then compared with the real thing:

- **Visual tokens per image.** Eleven preprocessing rules, from a fixed square to tiling to native resolution. We
  check them against ten real HF processors on twelve image sizes.
- **Where the parameters are.** The vision, connector and language parameters of 34 checkpoints, read from
  safetensors headers. No weights are downloaded.
- **The connector.** MLP, 2 x 2 merge, pixel shuffle, average pooling and a learned-query resampler. Each one is
  checked against its HF module. Its parameter formula is checked against 19 real checkpoints.
- **Positions.** Qwen's M-RoPE, Qwen2.5-VL's absolute-time video positions, Qwen3-VL's interleaved frequencies and
  InternVL3's V2PE. The first three are checked against HF code.
- **Tokens per page.** DeepSeek-OCR's question: how many text tokens does one vision token carry? We render real
  text at several font sizes and count.
- **Mixing and loss tips.** Square-root sampling with caps, samples against tokens, three SFT loss normalisations
  and DeepSeek-VL's modality warm-up.
- **Cross-family claims.** The post's "What correlates with capability" section, recomputed from its own Table 1.

Two pieces come from earlier tutorials. Pixel shuffle is imported from C7 (`c07-vlm/tinyvlm/projector.py`).
Temperature (square-root) mixture weights are imported from C8 (`c08-vlm-training/vlmtrain/mixture.py`). C7 also
builds and trains a full small VLM, and C8 trains it at scale. This lab adds the cross-family view.

## Layout

```
labs/how-the-open-vlm-families-were-built/
├── vlmfam/
│   ├── tokens.py       # preprocessing rules: fixed, tiling (AnyRes, InternVL, Idefics3, ...), native (Qwen, Pixtral)
│   ├── processors.py   # runs the real HF processors and counts their image tokens
│   ├── split.py        # parameter split from safetensors headers, by tensor name
│   ├── connectors.py   # connector modules and parameter formulas
│   ├── positions.py    # M-RoPE with video time, frequency interleaving, V2PE
│   ├── pages.py        # renders text pages; DeepEncoder token counts
│   ├── mixing.py       # caps, token shares, SFT loss normalisations, modality warm-up
│   └── report.py       # hardware line and result files
├── scripts/            # 01 to 08, one per post section, plus run_all.sh
├── data/families.json  # the post's Table 1, copied unchanged (used by script 08)
├── results/            # outputs of every script; each file starts with its hardware line and command
└── tests/
```

## Setup

From the repository root, with the repo's environment (`pip install -e ".[dev]"`). No extra packages.

Scripts 02, 03, 04 (part 3) and 06 need network access to the Hub. They download only small files: processor
configs, tokenizers, `config.json`, safetensors headers and WikiText-2 (about 1 MB). No weights. Gated Google and
Meta repos are read through their public `unsloth/` mirrors. No Hub token is needed.

The HF processors for Qwen2-VL, LLaVA-OneVision, InternVL and Pixtral also build a video processor. That needs
`torchvision`. This environment has none, so for those four the lab runs the PIL image processor and then the
processor's own counting step (see the docstring of `vlmfam/processors.py`). The other six run the full processor
and count image ids in `input_ids`.

## Run everything

```bash
bash labs/how-the-open-vlm-families-were-built/scripts/run_all.sh     # about 8 minutes on this machine, mostly Hub requests

# or one at a time
python labs/how-the-open-vlm-families-were-built/scripts/01_tokens_per_image.py    # offline
python labs/how-the-open-vlm-families-were-built/scripts/02_check_processors.py    # network
python labs/how-the-open-vlm-families-were-built/scripts/03_param_split.py         # network
python labs/how-the-open-vlm-families-were-built/scripts/04_connectors.py          # part 3 needs 03 and network
python labs/how-the-open-vlm-families-were-built/scripts/05_positions.py           # offline
python labs/how-the-open-vlm-families-were-built/scripts/06_tokens_per_page.py     # network
python labs/how-the-open-vlm-families-were-built/scripts/07_mixing_and_losses.py   # offline
python labs/how-the-open-vlm-families-were-built/scripts/08_cross_family.py        # offline

# tests
python -m pytest labs/how-the-open-vlm-families-were-built -m "not slow and not gpu and not network"
python -m pytest labs/how-the-open-vlm-families-were-built -m network
```

## Post sections and scripts

| Post section | Script | What it shows |
|---|---|---|
| Vision preprocessing: fixed, tiled or native (Figure 6) | `01_tokens_per_image.py`, `02_check_processors.py` | token counts per rule; rules equal to the real processors |
| Parameter split and the choice of encoder and decoder (Figure 7) | `03_param_split.py` | measured split for 34 checkpoints, next to the post's numbers |
| The connector; Token compression | `04_connectors.py` | four patterns at toy size and at real size |
| Architectural innovations: positions; Qwen-VL; Multiple images and video | `05_positions.py` | M-RoPE, time ids, interleaving, V2PE |
| DeepSeek-VL and DeepSeek-OCR (Figure 5) | `06_tokens_per_page.py` | text tokens per vision token on rendered pages |
| Data scale per stage and mixing; Training tips that recur | `07_mixing_and_losses.py` | sampling shares, token shares, loss weights |
| What correlates with capability | `08_cross_family.py` | the cross-family claims on the post's Table 1 |

## Results

Hardware for every number: AMD EPYC 7R13, 8 threads, torch 2.15.0, transformers 5.22.2. Nothing here depends on
speed. Every number is a count, a parameter total or a ratio, so it is the same on any machine.

### Rules against real processors (`results/02_check_processors.txt`)

| Measurement | Result |
|---|---|
| (repo, image size) pairs where the from-scratch rule equals the HF processor | 120 of 120 (10 repos x 12 sizes) |
| Letter scan, 1700 x 2200: LLaVA-1.5 / Gemma 3 / InternVL / Qwen2-VL / LLaVA-OneVision | 576 / 256 / 3,328 / 4,819 / 7,353 tokens |
| Phone photo, 4032 x 3024, Qwen2-VL (cap 16,384) | 15,552 tokens |
| Receipt, 600 x 2000, LLaVA-OneVision | 8,667 tokens: the post's "up to 7,290" counts tile tokens only, not the base image and row newlines |
| The post's simplified LLaVA-NeXT formula against the processor | from -2% (square photo) to +48% (slide): unpadding removes tokens |
| The post's simplified Idefics3 formula against the processor | -94% on a 300 px icon: the processor resizes the long side up to 1,456 px, so a small image costs 17 tiles (2,873 tokens), not one. The HF Idefics3 processor's limit is 1,456 px, not 1,820 |
| The post's Pixtral formula (row breaks ignored) | 2% to 5% below the processor |

### Parameter split (`results/03_param_split.txt`)

Billions of parameters, from safetensors headers. "Adapter" is the cross-attention layers inside the LLM.

| Model | Vision | Connector | Adapter | Language | Vision share | Post, Figure 7 |
|---|---|---|---|---|---|---|
| LLaVA-NeXT 7B | 0.304 | 0.021 | 0 | 6.739 | 4.3% | 4.3% |
| Qwen2-VL 72B | 0.631 | 0.068 | 0 | 72.706 | 0.9% | 0.9% |
| Qwen3-VL 235B-A22B | 0.416 | 0.160 | 0 | 235.094 | 0.2% | not in figure |
| InternVL3.5 38B | 5.536 | 0.092 | 0 | 32.762 | 14.4% | 14.4% |
| Molmo-72B | 0.291 | 0.310 | 0 | 72.707 | 0.4% | 0.4% |
| Idefics2 8B | 0.418 | 0.743 | 0 | 7.242 | 5.0% | not in figure |
| SmolVLM 256M | 0.086 | 0.007 | 0 | 0.163 | 33.7% | 40.8% |
| Llama 3.2 90B Vision | 0.864 | 0.063 | 17.113 | 70.554 | 1.0% | not in figure |

The full table has 34 checkpoints. Points worth knowing:

- Where the post's Figure 7 has a model, the measured vision share agrees to within 1 point for 13 of 15 models.
  The two exceptions are small models. SmolVLM-256M is 33.7% measured against 40.8% in the post: the checkpoint's
  vision tower has 86M parameters, not 93M, and its language side is 163M, which includes an untied 28M output
  head. Qwen2-VL 2B is 28.6% against 31.0%: the post counts the 34M merger with the encoder.
- Llama 3.2 90B is Llama 3.1 70B (70.55B, unchanged) plus 17.1B of cross-attention layers. The 11B is 8.03B plus
  1.75B. The adapter is about 20 times the vision encoder.
- The resampler is not small. Idefics2's perceiver has 652M parameters, more than its 418M encoder. Ovis2.5's visual
  embedding table and head hold 571M.
- Qwen2-VL's "675M" encoder in the post is the 631M ViT plus its 34M merger.

### Connectors (`results/04_connectors.txt`)

| Measurement | Result |
|---|---|
| From-scratch module against HF (LLaVA MLP, Qwen2-VL merger, InternVL projector, two pixel shuffles, Gemma 3 pool) | max abs difference 0 (Gemma 3: 2.9e-6) |
| Connector parameters from the config formula against the checkpoint | 19 of 19 exactly equal |
| One 448 px tile (1,024 SigLIP patches) into a 3,584-wide LLM: MLP / 2 x 2 merge / shuffle x4 / shuffle x16 / pool 4 x 4 / 64 queries | 1,024 / 256 / 256 / 64 / 64 / 64 tokens |

Idefics3 and InternVL both call their fold "pixel shuffle". They put the four neighbours in a different channel
order, so their weights are not interchangeable, but they hold the same values.

### Positions (`results/05_positions.txt`)

| Measurement | Result |
|---|---|
| `mrope_ids` against HF `get_rope_index` (Qwen2-VL and Qwen2.5-VL), random text/image/video sequences | 80 of 80 equal |
| Qwen3-VL interleaved frequency layout against HF | equal |
| Letter scan, Qwen2-VL: positions used in 1D / with M-RoPE | 4,839 / 99 |
| 60 s clip, time ids at 1 / 2 / 4 fps: Qwen2-VL | 0-29 / 0-59 / 0-119 |
| Same clip, Qwen2.5-VL (2 ids per second) | 0-116 / 0-118 / 0-119 |
| Four 13-tile images plus 400 text tokens: position range at V2PE step 1 / 1/4 / 1/16 | 13,712 / 3,729 / 1,233 |

### Tokens per page (`results/06_tokens_per_page.txt`)

WikiText-2 rendered on a 1,024 px page. Text tokens counted with DeepSeek-OCR's tokenizer. Vision tokens follow
from DeepEncoder's design: 64, 100, 256 and 400 for the 512, 640, 1,024 and 1,280 px modes.

| Font | Text tokens | Ratio at 64 / 100 / 256 / 400 vision tokens |
|---|---|---|
| 32 px | 399 | 6.2x / 4.0x / 1.6x / 1.0x |
| 20 px | 965 | 15.1x / 9.7x / 3.8x / 2.4x |
| 12 px | 2,549 | 39.8x / 25.5x / 10.0x / 6.4x |
| 10 px | 3,827 | 59.8x / 38.3x / 14.9x / 9.6x |

The same page size needs anything from 64 to 400 tokens to stay under a 10x ratio. Below 10x, the paper (as
quoted in the post) measured about 97% decoding precision. So the text density sets the budget, not the pixels.

### Mixing and losses (`results/07_mixing_and_losses.txt`)

| Measurement | Result |
|---|---|
| LLaVA-1.5 mix, share of the largest group (256K "other"): proportional / square root / cap 60K | 38.5% / 24.6% / 15.5% |
| 50:50 in samples, one-word answers against 200-token captions, share of supervised tokens | 0.5% / 99.5% |
| Gradient share of a 200-token answer in a batch of 1, 10 and 200 tokens: per token / per sample / square root | 94.8% / 33.4% / 77.3% |

### Cross-family claims (`results/08_cross_family.txt`)

| Claim in the post | On the post's Table 1 |
|---|---|
| Every model at or above 90 on DocVQA uses tiling or native resolution | holds: 16 tiling, 5 native, 1 cross-attention that also tiles (Llama 3.2), 0 fixed. Best fixed: Gemma 3, 86.6 |
| MMMU tracks the language model (hypothesis) | Spearman +0.64 with LLM size, but +0.81 with release month. All five benchmarks rank more closely with date than with size |
| RL stages and math (strong only within papers) | mean MathVista 71.1 with a preference or RL stage, 57.0 without; the RL rows are 7 months newer on average |

## Tests

AMD EPYC 7R13: 24 offline tests pass in about 40 s. The 2 `network` tests pass in about 45 s.
There are no `slow` or `gpu` tests. One offline test replays the recorded processor outputs in
`results/02_check_processors.json`, so the rules stay checked without a download.

## What was not measured

- No benchmark score in the post was reproduced. Training and evaluating these models is far beyond a lab. Every
  score in `data/families.json` is the post's, copied from the papers.
- No model weights were loaded, so nothing here runs a real VLM forward pass.
- DeepSeek-OCR decoding precision was not measured. Only the text-to-vision token ratio is computed.
- The DeepSeek-VL2 and MiniCPM-V slicing rules follow the post's description. Their processors need remote code, so
  they are not checked against a processor.
- Qwen-VL (2023) ships `.bin` weights, so its split is not read. Its 20% vision share in the post is not checked.
  PaliGemma has no public mirror and was skipped.
- The warm-up length and the V2PE step values in scripts 05 and 07 are illustrative, not taken from the papers.
- The response lengths in script 07, part 2, are the post's own example, not measured on a dataset.
