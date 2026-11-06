# Lab: The Cost of a Picture

A code companion for the post [The Cost of a Picture](https://srini-anirudh.github.io/blogs/the-cost-of-a-picture/).
The post asks where the cost of one image goes inside a vision-language model, and how much of it can be cut before
answers get worse. This lab makes that question runnable.

- `01_cost_model.py` recomputes the post's worked example from its formulas: token counts, encoder and prefill
  FLOPs, KV-cache bytes and decode time.
- `02_connectors.py` builds five connectors on a toy patch grid and shows what each one keeps.
- `03_stage_timing.py` measures the five stages of one request on Qwen2-VL-2B. Qwen2-VL uses 14-pixel patches and a
  2 x 2 merge, the rule behind the worked example, so its token counts can be checked against the cost model.
- `04_pruning.py` runs six token-reduction policies on Qwen2-VL-2B at the same budget, with a colour question and a
  small-text question.
- `05_prefix_cache.py` shows a prefix cache that keys only on token ids answering about the wrong picture, and the
  image-hash fix.

The connector pixel shuffle is imported from C7's `tinyvlm` instead of being rebuilt. C7
([`c07-vlm`](../../c07-vlm)) also builds a VLM from its parts and measures its forward pass; this lab adds the cost
accounting, token reduction and caching on top.

## Layout

```
labs/the-cost-of-a-picture/
├── pixelcost/
│   ├── costmodel.py    # token counts, FLOPs, KV bytes, decode time, concurrency (the post's formulas)
│   ├── connectors.py   # MLP, pixel shuffle (from C7), average pool, resampler, strided conv; the toy tasks
│   ├── selection.py    # random, uniform grid, pooling, attention rank, diversity, ToMe merging
│   ├── qwen.py         # Qwen2-VL split into preprocess, encode, reduce, prefill, decode; FastV's score
│   ├── serving.py      # image hash, block prefix cache, embedding cache
│   ├── synth.py        # synthetic pictures with a coloured shape and a small printed number
│   └── testing.py      # a toy-size random Qwen2-VL for the offline tests
├── scripts/            # 01 to 05, and run_all.sh
├── results/            # outputs written by the scripts; first line hardware, second line command
└── tests/
```

## Setup

From the repository root, with the repo's environment (`pip install -e ".[dev]"`). No extra packages. Scripts 03,
04 and 05 download [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), about 4.5 GB.
The stock Qwen2-VL processor also wants torchvision for video, so `qwen.Processor` does the image part by hand: it
renders the chat, runs the PIL image processor, and expands one `<|image_pad|>` into one placeholder per visual
token.

## Run everything

```bash
python -m pytest labs/the-cost-of-a-picture -m "not slow and not gpu and not network"   # 35 tests, about 20 s
python -m pytest labs/the-cost-of-a-picture -m "network or slow"                        # 4 tests, downloads Qwen2-VL

python labs/the-cost-of-a-picture/scripts/01_cost_model.py            # instant
python labs/the-cost-of-a-picture/scripts/02_connectors.py            # about 2 min on 8 CPU threads
python labs/the-cost-of-a-picture/scripts/03_stage_timing.py --device cpu --sizes 448x448,672x672 --decode-steps 16 --repeats 2
python labs/the-cost-of-a-picture/scripts/05_prefix_cache.py --device cpu   # about 1 min

# On one CUDA GPU.
python labs/the-cost-of-a-picture/scripts/03_stage_timing.py --device cuda --eager --repeats 5
python labs/the-cost-of-a-picture/scripts/04_pruning.py --device cuda --n 48

bash labs/the-cost-of-a-picture/scripts/run_all.sh [cpu|gpu|all]   # regenerates results/; set PY to pick the Python
```

`04_pruning.py --device cpu --n 16` also runs on a CPU, in tens of minutes. Use `--n`,
`--budgets` and `--policies` to shorten it.

## Results

Every number names its hardware. Raw outputs are in `results/`; each file starts with a hardware line and the command.
The CPU is AMD EPYC 7R13, 8 threads, torch 2.15.0, FP32. The machine was shared with other jobs, so the Qwen2-VL
timings are noisy (see the notes under the table). Token counts, KV bytes and accuracies do not depend on load.

Tests on AMD EPYC 7R13, 8 threads: 35 passed offline (`not slow and not gpu and not network`), 4 passed with
`network or slow`. No `gpu` tests.

| Measurement | Result | Hardware |
|---|---|---|
| Post's worked example, recomputed from the formulas (`01`) | 6,912 patches, 1,728 tokens, 9.3 TFLOP encoder, 26.0 TFLOP prefill, 56 KiB/token, 101 MB KV (image 98%), 710 MB without GQA, 4.21 ms per decode step. Every line matches the post | none (arithmetic), H100 datasheet constants |
| Decode vs image cost, same example (`01`) | image share of request time 93% at a 1-token answer, 62% at 10, 8% at 200 | estimate, not measured |
| Requests that fit in 80 GB next to 7B weights, 200-token answers (`01`) | 549 at 100% of visual tokens, 979 at 50%, 1,608 at 25% | estimate, not measured |
| Qwen2-VL-2B token count vs `costmodel` (`03`, tests) | 256 / 576 / 1,728 visual tokens for 448 / 672 / 1008x1344 images: identical | CPU |
| Qwen2-VL-2B KV cache, bytes measured vs formula (`03`) | 16.5 MB and 34.9 MB for the 448 and 672 pictures: identical | CPU, FP32 |
| Connectors, toy glyph task, chance 0.25 (`02`, mean of 3 seeds) | MLP (64 tokens) 1.00, pixel shuffle (16) 1.00, resampler (4) 1.00, strided conv (16) 1.00, average pool (16) **0.23** | CPU |
| Connectors, toy blob task (`02`) | all five connectors 0.999 to 1.00 | CPU |
| Qwen2-VL-2B, 576 tokens, colour / number accuracy, 48 pictures (`04`) | full: 1.00 / 1.00 | NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2, BF16 |
| Same, 144 tokens kept (colour / number) | random 1.00 / 0.90, uniform 1.00 / 0.92, pool 1.00 / 0.75, attention 1.00 / 0.94, diversity 1.00 / 1.00, ToMe 1.00 / 0.92 | same H100 |
| Same, 64 tokens kept | random 1.00 / 0.42, uniform 0.98 / 0.12, pool 1.00 / 0.04, attention 1.00 / 0.44, diversity 1.00 / 0.85, ToMe 1.00 / 0.42 | same H100 |
| Same experiment, 16 pictures, 64 tokens kept | random 0.38, uniform 0.06, pool 0.00, attention 0.25, diversity 0.75, ToMe 0.69 (number accuracy) | CPU, FP32 |
| Layer-2 last-token attention by image row (`04`) | 66.5% of the mass in the top 12 of 24 rows; most of it in the first 4 and last 3 rows | H100 (CPU run: 66.6%) |
| Prefill, 576 vs 144 vs 64 visual tokens (`04`) | H100: 24.2, 24.0, 24.2 ms (flat). CPU: 3.3 s, 1.25 s, 0.86 s | H100 BF16; CPU FP32, noisy |
| Stage times, Qwen2-VL-2B, 448x448 (`03`) | preprocess 9 ms, encoder 3.2 s, prefill 3.1 s, decode 0.37 s per token | CPU, noisy |
| Stage times, 672x672 (`03`) | preprocess 14 ms, encoder 6.6 s, prefill 2.8 s, decode 0.32 s per token | CPU, noisy |
| Prefix cache with token-id keys only (`05`) | picture B (blue square) reuses 272 of 288 tokens of picture A and answers **"red"** | CPU |
| Prefix cache with the image hash in the key (`05`) | picture B reuses 0 tokens and answers "Blue"; a follow-up about A reuses 272 of 291 tokens and skips the encoder | CPU |
| Stage times on the GPU, Qwen2-VL-2B, 448x448 / 672x672 / 1344x1008 / 1792x1344 (`03 --device cuda --eager`) | encoder 23.0 / 24.1 / 59.8 / 121.3 ms; prefill 24.5 / 25.1 / 26.8 / 27.3 ms; decode about 22 ms per token at every size | NVIDIA H100 80GB HBM3, BF16, median of 5 |
| Image share of request time on the GPU, 1-token / 10-token / 200-token answers | 448x448: 71% / 20% / 1%; 1792x1344: 92% / 52% / 5% | same run |
| Eager vs SDPA prefill, same FLOPs | ratio 1.16 / 1.15 / 1.32 / 3.23 for the four sizes (1792x1344: 88.1 ms eager vs 27.3 ms SDPA) | same run |

Notes on the numbers.

- The pruning run has 48 pictures, 24 with the number in the top band and 24 in the bottom band, so one miss moves
  a number-accuracy cell by 0.02. At 144 tokens only pooling (0.75) stands clearly below the rest. At 64 tokens
  the policies separate: diversity 0.85, then attention, random and ToMe near 0.43, then uniform 0.12 and pooling
  0.04. The ordering of random against attention against ToMe is within noise. Random selection beats uniform and
  pooling here, which is the post's point that simple baselines are hard to beat. Scoring tokens by attention did
  not beat random (0.44 against 0.42). Diversity did.
- On the H100 the prefill time does not change with the token budget (about 24 ms for 64, 144 and 576 visual
  tokens). A 2B model on 600 tokens is bound by kernel launches, not by FLOPs, so token pruning saves compute that
  this GPU was not waiting on. The CPU run shows the FLOP-bound case (3.3 s down to 0.86 s).
- The attention policy does not simply drop the top of the image here. Layer-2 attention is large at both
  edges (first 4 rows and last 3 rows hold most of it), and the kept tokens' mean row is 9.8 at 144 tokens and 7.9
  at 64: a pull towards the top, not the bottom. Top-band and bottom-band numbers are recovered at similar rates
  (0.92 and 0.96 at 144 tokens). The post's claim of a bottom-heavy bias comes from LLaVA-1.5 and is not reproduced
  on Qwen2-VL by this run.
- The first CPU stage-timing run had a decode step of 2.5 s per token because the machine was loaded. The rerun in
  `results/` took 0.32 to 0.37 s. The 672x672 prefill (2.8 s) is faster than the 448x448 prefill (3.1 s), which
  shows the noise. Use these as orders of magnitude.
- `01` prints the 3x3 and 4x4 merge columns with floor division. Real connectors pad the grid first.

## Post sections and scripts

| Post section | Script or module |
|---|---|
| What one picture costs (provider token rules) | `01_cost_model.py`, `costmodel.py` |
| Where the cost goes: counting tokens, compute, KV memory, worked example | `01_cost_model.py`; measured on a real model by `03_stage_timing.py` |
| Fewer tokens by design (pixel shuffle, pooling, resampler, convolution) | `02_connectors.py`, `connectors.py` |
| Pruning and merging after the fact (FastV score, random, uniform, diversity, ToMe) | `selection.py`, `qwen.py`, `04_pruning.py` |
| What breaks when tokens go (position bias, random and pooling baselines, reading vs scene) | `04_pruning.py` |
| Insight: FLOPs vs time, fused attention kernels | `03_stage_timing.py --eager` (GPU, `results/03_stage_timing_cuda.txt`); `qwen.attention_received` scores tokens without the attention matrix |
| Serving images at scale: embedding cache, image-hash prefix cache | `serving.py`, `05_prefix_cache.py` |
| Measuring efficiency honestly (include every stage, random and pooling baselines) | `03_stage_timing.py`, `04_pruning.py` |

## What was not measured

- The papers' own results (FastV, PyramidDrop, SparseVLM, VisionZip, DART, Gemma 3 Pan and Scan, the
  inference-optimal scaling fit, EPD, ModServe) were not reproduced. This lab shows each mechanism small and
  measures a real 2B model on synthetic pictures only.
- The pruning policies act on the language model's input. FastV prunes after layer 2, PyramidDrop in stages inside
  the model. The attention policy uses FastV's score but applies the kept set at the input, so it is not FastV.
  DART, SparseVLM, VisionZip and CDPruner are not implemented. The diversity policy is farthest-point selection in the
  style of DivPrune.
- Accuracy is on 48 synthetic pictures with two question types. It says nothing about MME, POPE, TextVQA or RefCOCO.
  Grounding was not tested.
- The connector comparison is a toy (8x8 grid of 8-dim patches, one transformer layer). It shows what a block average
  discards and what pixel shuffle keeps. It does not rank connectors in real VLMs; the post's MM1 point (connector type
  matters less than token count) is not tested.
- Quantization, distillation, speculative decoding, video, disaggregation and scheduling are not built. The post
  covers them from the literature.
- The GPU stage timing is one run of 5 repeats on a shared host (`results/03_stage_timing_cuda.txt`). Real batched serving, TTFT under load
  and concurrency were not measured. The concurrency figures in `01` are estimates from the formula.

