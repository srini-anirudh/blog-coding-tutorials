# Lab: The VLM Architecture Gallery

The gallery draws 71 vision-language models on one plate and argues that each is a set of answers to five
questions: how the image gets in, what compresses the patches, where vision meets the language model, what is
frozen, and what a picture costs. This lab builds the few recurring answers as small swappable PyTorch modules and
puts them side by side.

Theory post: [The VLM Architecture Gallery](https://srini-anirudh.github.io/blogs/the-vlm-architecture-gallery/).

Part of [blog-coding-tutorials](https://github.com/srini-anirudh/blog-coding-tutorials). Two tutorials go deeper
on single paths through the gallery. [C7](../../c07-vlm/) builds one LLaVA-style model (SigLIP, pixel shuffle, MLP,
SmolLM2) and its processor end to end. [C8](../../c08-vlm-training/) trains it in stages. This lab does not rebuild
those. It builds the alternatives and compares them.

## What you build

| Module | What it holds |
|---|---|
| `gallery/connectors.py` | Six connectors with one contract, `forward(x, grid)` and `out_tokens(h, w)`: linear, MLP, pixel shuffle, average pooling, attentional pooling (Molmo-like), Perceiver resampler. |
| `gallery/fusion.py` | A tiny GPT whose projections are `RoutedLinear` layers, plus the edits each fusion point makes to a pretrained LM: tanh-gated cross-attention (Flamingo), a full visual expert (CogVLM), low-rank visual deltas (PLoRA). Also a rough attention-cost model. |
| `gallery/model.py` | Frozen patch encoder, connector, fusion and LM put together. The encoder-free mode (Fuyu) skips the encoder and projects raw patches. |
| `gallery/inputs.py` | Visual-token counts for fixed squares, LLaVA-NeXT anyres tiles, InternVL dynamic tiles, Qwen2-VL native resolution and overlapping multi-crop. |
| `gallery/cost.py` | KV bytes per visual token, the post's three serving cards, and a measured Hugging Face cache to check the formula. |
| `gallery/data.py`, `gallery/train.py` | Synthetic glyph pages, text pretraining of the tiny LM, freezing schedules, training and evaluation. |
| `gallery/hub.py` | Reads nine real configs from the Hugging Face Hub and computes visual tokens and KV cost from their fields. |

The toy task. A page is 16 x 16 pixels: a 4 x 4 grid of cells, each holding one of 8 glyphs plus noise. One patch
is one cell, so the encoder sees 16 patches. Two questions pull the connectors apart. The **fine** question
("[CELL] r c": which glyph is in row r, column c?) needs every cell kept apart, like reading small text. The
**coarse** question ("[MAJ]": which glyph fills half the page?) only needs a summary, like naming the main object
in a photo. Chance is 0.125 for both. The tiny LM (4 layers, width 64, 205,568 parameters) is first pretrained on
text-only versions of both questions, where the page arrives as 16 glyph tokens. It also learns one unrelated text
task, addition mod 8, which no visual training touches. That task shows forgetting.

## Setup

From the repository root, with the repo's environment (`pip install -e ".[dev]"`). No extra packages.
`07_hub_configs.py` and the `network` test download a few kB of `config.json` files. Gemma 3 is read from the public
`unsloth/gemma-3-4b-it` copy, because `google/` repos are gated. On a shared machine set `OMP_NUM_THREADS`.
The scripts default to 4 threads.

## Run everything

```bash
bash labs/the-vlm-architecture-gallery/scripts/run_all.sh        # set PY to choose the Python

# or one at a time
python labs/the-vlm-architecture-gallery/scripts/01_connectors.py      # seconds
python labs/the-vlm-architecture-gallery/scripts/02_input_tokens.py    # seconds
python labs/the-vlm-architecture-gallery/scripts/03_fusion_identity.py # pretrains the tiny LM once (cached in runs/)
python labs/the-vlm-architecture-gallery/scripts/04_kv_cost.py         # seconds
python labs/the-vlm-architecture-gallery/scripts/05_connector_bench.py --seed 0   # 7 trainings
python labs/the-vlm-architecture-gallery/scripts/05_connector_bench.py --seed 1
python labs/the-vlm-architecture-gallery/scripts/05_connector_bench.py --report   # merge seeds
python labs/the-vlm-architecture-gallery/scripts/06_fusion_freeze.py --seed 0     # 6 trainings
python labs/the-vlm-architecture-gallery/scripts/06_fusion_freeze.py --seed 1
python labs/the-vlm-architecture-gallery/scripts/06_fusion_freeze.py --report
python labs/the-vlm-architecture-gallery/scripts/07_hub_configs.py     # needs network
python labs/the-vlm-architecture-gallery/scripts/08_xattn_longer.py    # gated xattn at 2.5k, 5k and 10k steps

# tests
python -m pytest labs/the-vlm-architecture-gallery -m "not slow and not gpu and not network"
python -m pytest labs/the-vlm-architecture-gallery -m network
```

Each script writes `results/<name>.txt` (and `.json`), starting with the hardware and the command.

## Post sections and scripts

| Post section | Script | What you see |
|---|---|---|
| Four ways to join two towers (Figure 1) | `01_connectors.py`, `05_connector_bench.py` | Linear, resampler, pixel shuffle and merge on one grid: tokens out, parameters, what each keeps. |
| Question 01, getting in | `02_input_tokens.py` | Tokens for one picture under fixed, tiled, native and multi-crop input. |
| Question 02, compressing | `01_connectors.py`, `05_connector_bench.py` | Shuffle is lossless and pooling is not. Accuracy on fine and coarse questions per connector. |
| Question 03, joining | `03_fusion_identity.py`, `06_fusion_freeze.py` | Gated xattn is an exact copy of the LM at init. Expert parameter cost. Accuracy per fusion point; xattn learns detail more slowly. |
| Question 04, freezing | `06_fusion_freeze.py`, `08_xattn_longer.py` | Frozen LM keeps its text skills exactly. An unfrozen LM forgets without text replay. |
| Question 05, paying | `04_kv_cost.py` | The post's KV arithmetic, checked against a real cache. |
| What a picture costs (Figure 3), the fact sheets | `02_input_tokens.py`, `07_hub_configs.py` | Token counts from nine real Hub configs against the gallery's numbers. |

## Results

Hardware for every number below: **AMD EPYC 7R13 CPU**, float32, torch 2.15.0, on a shared machine under heavy
load. Thread counts are in each results file: 4 per process for the training runs (two processes at a time), 8 or
2 for the others. Training times are wall-clock under that load and are only a rough guide.

### Connectors: tokens, parameters, what they keep (`01_connectors.py`, `results/01_connectors.txt`)

A 32 x 32 patch grid (448 px at patch 14), random features of width 1024, into a width-2048 model.

| Connector | Tokens out | Ratio | Parameters |
|---|---|---|---|
| Linear | 1024 | 1x | 2,099,200 |
| MLP | 1024 | 1x | 6,295,552 |
| Pixel shuffle (2 x 2) + MLP | 256 | 4x | 12,595,200 |
| Average pool (2 x 2) + MLP | 256 | 4x | 6,295,552 |
| Attentional pool (2 x 2) | 256 | 4x | 10,493,952 |
| Resampler, 64 queries | 64 | 16x | 14,774,272 |

The resampler returns 64 tokens on 16 x 16, 32 x 32 and 48 x 64 grids. Pixel shuffle returns 64, 256 and 768.
Pixel shuffle is inverted exactly (maximum error 0). Swapping two patches inside every 2 x 2 window changes the
average-pool output by 1.5e-07 and the pixel-shuffle output by 0.93.

### Which connector keeps fine detail (`05_connector_bench.py`, `results/05_connector_bench.txt`)

Toy task, 16 patches per page, 2,500 steps, two seeds, mean (min to max) on 2,048 fresh pages. Chance is 0.125.
AMD EPYC 7R13, 4 threads per process, two processes at a time, shared machine.

| Connector | Tokens | Fine: glyph at row r, column c | Coarse: majority glyph |
|---|---|---|---|
| Linear | 16 | 0.999 (0.999-0.999) | 1.000 |
| MLP | 16 | 0.999 (0.999-1.000) | 1.000 |
| Resampler, 16 queries | 16 | 0.941 (0.929-0.954) | 1.000 |
| Pixel shuffle, 2 x 2 | 4 | 0.816 (0.803-0.829) | 1.000 |
| Attentional pool, 2 x 2 | 4 | 0.404 (0.400-0.408) | 1.000 |
| Average pool, 2 x 2 | 4 | 0.399 (0.395-0.403) | 1.000 |
| Resampler, 4 queries | 4 | 0.372 (0.369-0.375) | 1.000 |

All seven keep the coarse summary. They differ only on the fine question. At four tokens, pixel shuffle keeps far
more than pooling or four queries, which agrees with the post's claim that folding beats averaging for detail.
Attentional pooling did not beat plain averaging here. We did not test why. One guess: its query is the window mean, so it has little to select
with. Sixteen queries recover most of the detail, as expected when the queries equal the patch count.

### Fusion points and freezing (`06_fusion_freeze.py`, `results/06_fusion_freeze.txt`)

Same pretrained LM (text accuracy 1.000 on all three text tasks), MLP connector, 16 visual tokens, 2,500 steps,
two seeds, no text replay. Same hardware as above.

| Recipe | Trainable params | Fine | Coarse | Text lookup after | Text addition after |
|---|---|---|---|---|---|
| Prefix, LM frozen | 8,320 | 0.997 | 1.000 | 1.000 | 1.000 |
| Prefix, LM trains | 213,888 | 0.999 | 1.000 | 1.000 | 0.675 (0.671-0.679) |
| Gated cross-attention, LM frozen | 208,264 | 0.290 (0.242-0.338) | 0.998 | 1.000 | 1.000 |
| Visual expert, LM frozen | 207,232 | 0.999 | 1.000 | 1.000 | 1.000 |
| PLoRA r = 8, LM frozen | 41,088 | 0.998 | 1.000 | 1.000 | 1.000 |
| Encoder-free, LM trains | 206,656 | 0.999 | 1.000 | 0.865 | 0.087 (0.032-0.141) |

Frozen-LM recipes keep text skills exactly. Unfreezing the LM and training on visual questions alone costs the
unrelated addition task a third of its accuracy, and the encoder-free model, which trains the whole stack on
pixels, forgets it almost entirely. Gated cross-attention learned the coarse question but was still far behind on
the fine one at 2,500 steps. `08_xattn_longer.py` (one seed, each row its own run) shows it is slow, not unable:

| Steps | Fine | Coarse |
|---|---|---|
| 2,500 | 0.338 | 1.000 |
| 5,000 | 0.901 | 1.000 |
| 10,000 | 0.998 | 1.000 |

Structure at init (`03_fusion_identity.py`): gated cross-attention with an image attached reproduces the frozen
LM's logits exactly (difference 0). A full visual expert adds 198,912 parameters to a 205,568-parameter LM
(0.97x). PLoRA r = 8 adds 32,768 (0.16x). For a real-size picture (576 visual and 64 text tokens, width 4096,
32 layers) the attention-score cost is 107.4 G multiply-accumulates as a prefix and 10.7 G with cross-attention,
with the same number of cached K/V positions (formula, `gallery/fusion.py`, not a timing).

### Tokens per picture under each input strategy (`02_input_tokens.py`, `results/02_input_tokens.txt`)

| Strategy | 1024 x 768 | Phone photo 4032 x 3024 | Receipt 600 x 2000 | A4 page, 150 dpi |
|---|---|---|---|---|
| Fixed 336 (LLaVA-1.5) | 576 | 576 | 576 | 576 |
| Anyres tiles (LLaVA-NeXT, 5 canvases) | 2,340 | 2,340 | 2,232 | 2,256 |
| 448 tiles, up to 12 + thumbnail (InternVL 2.5) | 3,328 | 3,328 | 1,024 | 1,792 |
| Native, 2 x 2 merge (Qwen2-VL, default limits) | 999 | 15,552 | 1,491 | 2,772 |
| Overlapping 336 crops, 56 px overlap, 144 tokens each | 1,872 | 1,872 | 720 | 1,872 |

With up to 40 tiles an A4 page uses 9,216 tokens and the 40-tile maximum is 10,496. The tests check the LLaVA-NeXT
count, the Qwen2-VL resize and the tile grid against the Hugging Face code. The last row is a generic rule, not one
model's recipe.

### KV cost of a picture (`04_kv_cost.py`, `results/04_kv_cost.txt`)

32 layers, 8 KV heads, head dim 128, bf16: 128 KiB per token. "Measured" is the cache of a real
`LlamaForCausalLM` with that attention shape on AMD EPYC 7R13 (it matches the formula to the byte).

| Setting | Tokens | KV MiB, formula | KV MiB, measured | Images in 128K context |
|---|---|---|---|---|
| 336 fixed | 576 | 72.0 | 72.0 | 222 |
| 2 x 2 tiles + global | 2,880 | 360.0 | 360.0 | 44 |
| Native 1024 | 1,300 | 162.5 | 162.5 | 98 |
| 40-tile page | 10,240 | 1280.0 | 1280.0 | 12 |

### Real configs from the Hub (`07_hub_configs.py`, `results/07_hub_configs.txt`)

Config files only, no weights. Tokens are computed from config fields. KV is for the visual tokens in the middle
column, using the model's own language-model shape in bf16.

| Model | Tokens from config | The post says | KV KiB per token | KV MiB |
|---|---|---|---|---|
| LLaVA-1.5 7B | 576 | 576 | 512 | 288.0 |
| LLaVA-NeXT (Mistral 7B) | 2,928 | up to 2,880 | 128 | 366.0 |
| IDEFICS2 8B | 64 | 64 | 128 | 8.0 |
| IDEFICS3 8B | 169 per tile | 169 per tile | 128 | 21.1 |
| InternVL 2.5 1B | 256 per tile | 256 per tile | 12 | 3.0 |
| Qwen2-VL 2B | up to 16,384 | up to 16,384 | 28 | 448.0 |
| Pixtral 12B | up to 4,096 | up to 4,096 | 160 | 640.0 |
| CogVLM | 1,225 | 1,225 | 512 | 612.5 |
| Gemma 3 4B | 256 per window | 256 per window | 136 | 34.0 |

Eight of nine agree. LLaVA-NeXT differs by 48: the post's "2,880" is five 576-token views. Hugging Face's
implementation removes padding rows but adds one newline token per row, and the largest canvas gives 2,928. Two of
the checks carry caveats. Pixtral's 4,096 counts patches at the largest size and ignores the row-break tokens. The
Gemma 3 cache figure assumes full attention and overstates a long context, since most layers use a sliding window.
The KV column also shows how much the language-model shape matters: the same 576 tokens cost 288 MiB on LLaVA-1.5
7B (32 KV heads) but 72 MiB with 8 KV heads.

### Tests

`python -m pytest labs/the-vlm-architecture-gallery -m "not slow and not gpu and not network"`: 77 passed in about
21 s (AMD EPYC 7R13, 2 threads). `-m network`: 1 passed in about 17 s.

## What was not measured

- No paper's benchmark is reproduced. The connector and fusion comparisons use a 0.2M-parameter LM and synthetic
  glyphs. They show the mechanisms the post describes. They do not show the size of any effect in a real VLM.
- The stand-in encoder is a frozen random per-patch MLP. It has no attention across patches, so a connector cannot
  rely on the encoder having spread information around. Real ViTs do spread it, which can soften pooling losses.
- No vision encoder or VLM checkpoint is run. The Hub check reads configs only.
- PaliGemma is not checked: its configs are gated and we found no public copy we trust.
- Serving speed, packing and cross-rank load balancing for native resolution are discussed in the post but not
  measured here. C8 measures packing.
- No GPU was used. Nothing in the lab needs one.
