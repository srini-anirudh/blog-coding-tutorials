# Lab: How Vision-Language Models Scale

Runnable code for the post [How Vision-Language Models Scale](https://srini-anirudh.github.io/blogs/how-vision-language-models-scale/).
The post asks what a scaling law means when a model has two towers, a connector, and a variable number of tokens
per picture. This lab builds the post's central mechanisms in small, readable code and runs them.

- **The budget of one sample.** The post's FLOP accounting, with the encoder share and the frozen-tower rule.
- **The 6 / 4 / 2 rule, measured.** PyTorch's FLOP counter runs on a real two-tower model and gives k per tower.
- **Resolution and attention.** Both towers grow with the square of the image side. Attention grows with its fourth power.
- **The compute ladder.** The post's Table 2, recomputed from what each paper reports.
- **Quality against quantity.** The utility-decay model, and a real run in which the best filter changes with the budget.
- **The law forms.** Chinchilla allocation, a fit check, the fusion splits, the modality interaction term and the
  inference-optimal trade-off.
- **Visual tokens against LLM size.** Real small VLMs trained with 1 to 64 visual tokens and three LLM widths.
- **What freezing saves on a GPU.** Step time and peak memory of a 1.1B-parameter two-tower model.

None of the paper-scale results in the post were reproduced. Where the post quotes a large study, the lab shows the
mechanism at small scale or recomputes the post's arithmetic.

## Layout

```
labs/how-vision-language-models-scale/
├── vlmscale/
│   ├── budget.py    # patches, visual tokens, per-sample FLOPs with k per tower, encoder share, attention FLOPs
│   ├── model.py     # TwoTowerVLM: ViT -> pixel shuffle or pooling -> MLP connector -> causal LLM
│   ├── flops.py     # per-tower FLOPs of a training step with FlopCounterMode, under four freezing settings
│   ├── ladder.py    # Table 2: 6ND and chip-hours x peak x 40% estimates
│   ├── laws.py      # Chinchilla, closed-form allocation, Huber fit, interaction term, inference trade-off, utility decay
│   ├── quality.py   # noisy pool with quality scores, filtering, student training
│   ├── data.py      # glyph-board task: a coarse question and a reading question
│   ├── train.py     # training loop with loss against training FLOPs
│   └── hw.py        # hardware line for every results file
├── scripts/         # 01 to 07, listed below, plus run_all.sh
├── tests/           # pytest, all offline
└── results/         # real outputs; each file starts with the hardware and the command
```

The pixel-shuffle merge is the same operation as C7's (`c07-vlm/tinyvlm/projector.py`). A test checks that the two
give identical outputs. C7 and C8 build and train a full VLM from pretrained parts. This lab does not repeat that.
It looks at what the post adds: where the compute goes, and how the budget should be split.

## Setup

From the repository root, after `pip install -e ".[dev]"`. Nothing downloads. Set `OMP_NUM_THREADS` on a shared machine.

## Run everything

```bash
# Tests (about 20 s on CPU).
python -m pytest labs/how-vision-language-models-scale -m "not slow and not gpu and not network"

# Scripts. Each prints its tables and writes results/<name>.txt (and .json where useful).
python labs/how-vision-language-models-scale/scripts/01_budget.py            # seconds
python labs/how-vision-language-models-scale/scripts/02_measure_k.py         # seconds
python labs/how-vision-language-models-scale/scripts/03_compute_ladder.py    # instant
python labs/how-vision-language-models-scale/scripts/04_quality_quantity.py  # about 12 min on a busy CPU (--quick: about 1 min)
python labs/how-vision-language-models-scale/scripts/05_laws.py              # about 10 s
python labs/how-vision-language-models-scale/scripts/06_tokens_vs_size.py    # about 30 min on a busy CPU (--quick: 2 short runs)
python labs/how-vision-language-models-scale/scripts/07_frozen_step_gpu.py   # needs CUDA; --small runs a CPU smoke test

PY=python bash labs/how-vision-language-models-scale/scripts/run_all.sh     # all of the above
```

## Which script covers which section of the post

| Post section | Script |
|---|---|
| An image does not have a fixed size in tokens | `01_budget.py` (resolution table) |
| The compute budget of one training sample, Figure 1 | `01_budget.py`, `02_measure_k.py` |
| Insight: the encoder share scales with the merge factor | `01_budget.py`, `02_measure_k.py` |
| Nuance: a frozen LLM saves a third, not two thirds | `02_measure_k.py`, `07_frozen_step_gpu.py` |
| An interactive budget, Figure 2, and the hidden attention cost | `01_budget.py` |
| Model size versus resolution at equal cost (PaliGemma 2) | `01_budget.py` |
| Scaling laws in one page | `05_laws.py` parts 1 and 2 |
| Utility decays with repetition, Figure 4 | `04_quality_quantity.py` |
| Competition and synergy between modalities, Figure 5 | `05_laws.py` part 4 |
| Early fusion versus late fusion, Figure 6 | `05_laws.py` part 3 |
| An inference-optimal law, Figure 8 | `05_laws.py` part 5, `06_tokens_vs_size.py` |
| What open VLMs actually spend, Table 2 and Figure 9 | `03_compute_ladder.py` |

## Results

Every number names its hardware. Raw output is in `results/`, under the name of the script that wrote it. The CPU was
shared with other jobs, so timings on it are rough. Accuracies and FLOP counts do not depend on load.

### Arithmetic and FLOP counts (AMD EPYC 7R13, 8 threads, torch 2.15.0)

These do not depend on hardware. They check the post's numbers and the k rule.

| Measurement | Result | Post says |
|---|---|---|
| Figure 1: encoder share, 0.4B encoder, 448 px, 2x2 merge, 200 text tokens, 7B LLM | 11.4% | about 11% |
| The same with a 0.5B LLM | 64.2% | about 64% |
| Encoder share, Qwen2-VL-2B / 7B / 72B (text ignored) | 64.3% / 26.2% / 3.6% | about 64%, under 4% |
| SmolVLM-256M encoder share, r = 4 / r = 2 | 91.7% / 73.4% | about 73% with r = 2 |
| LLM-equivalent of a 0.675B encoder with a 2x2 merge | 2.70B | 2.7B |
| Measured k, all towers trained: encoder / connector / LLM | 5.96 / 6.00 / 6.00 | 6 |
| Measured k, frozen encoder | 2.00 | 2 |
| Measured k, frozen LLM above a trained connector | 4.00 | 4 |
| LLM attention-score FLOPs, trained vs frozen LLM | 113.2 vs 113.2 MFLOP (unchanged) | not stated |
| Encoder share in the counted model: formula / measured | 42.4% / 42.2% | |
| Encoder attention share at 1344 px, full / windowed attention (Qwen2.5-VL-like) | 53% / 13% | "no longer small" |
| PaliGemma 2 relative cost, 3B @ 896 px vs 28B @ 224 px: FLOP model (with attention) | 16.2 vs 9.3 | paper: 23.5 vs 18.9 |
| Table 2: all 13 estimates recomputed from the reported inputs | all 13 match to one significant figure | |
| InternVL2.5-78B multimodal training / Qwen2.5-72B text pretraining | 0.72% | under 1% |
| Qwen2.5-VL-72B pretraining / Qwen2.5-72B text pretraining | 23% | roughly a quarter |

The encoder's measured k is 5.96, not 6. Its patch-embedding layer never needs a gradient for the pixels. The
PaliGemma 2 row agrees with the paper's direction (the small model at high resolution costs more). It does not
match the paper's numbers, which are measured TPU cost rather than FLOPs.

### Quality against quantity, real run (AMD EPYC 7R13, 2 threads, torch 2.15.0)

A pool of 100,000 points labelled by a teacher MLP, with label noise that falls as a quality score rises. Clean test
accuracy, mean of 3 seeds (`results/quality_quantity.txt`):

| Sample views | Top 10% (10K kept, 3% noise) | Top 30% (30K kept, 8% noise) | No filter (100K, 27% noise) | Best |
|---|---|---|---|---|
| 40,000 | **0.722** | 0.718 | 0.707 | top 10% |
| 160,000 | 0.720 | **0.735** | 0.728 | top 30% |
| 640,000 | 0.682 | 0.738 | **0.743** | no filter |
| 2,560,000 | 0.677 | 0.671 | **0.748** | no filter |

The winner moves from the aggressive filter to the mild one to no filter as the budget grows. This held in each of
the 3 seeds at these four budgets. At 10,000 views the three are within 1.2 points and the order changes between
seeds. The margins are small, from 0.4 to 7 points. Unlike Goyal et al.'s model, the filtered runs lose accuracy
when repeated for many epochs: the small student memorises the kept set. The utility-decay model in part 1 of the
same script uses illustrative constants and shows the same order of winners.

### Visual tokens against LLM size, real run (AMD EPYC 7R13, 6 jobs x 2 threads, torch 2.15.0)

24 small VLMs trained from scratch, 1,500 steps of batch 128 each, 2 seeds. The encoder is the same in every run
(101K matmul parameters, 64 patches). Its output is average-pooled to V tokens. Accuracy on 1,024 fresh images per
question; chance is 0.125 (`results/tokens_vs_size.txt`):

| V | LLM width (N_L) | Inference cost N_L (Q+V) | Coarse question | Reading question |
|---|---|---|---|---|
| 1 | 32 (26K) | 0.08M | 1.000 | 0.138 |
| 1 | 128 (398K) | 1.19M | 1.000 | 0.130 |
| 4 | 64 (101K) | 0.60M | 1.000 | 0.346 |
| 16 | 32 (26K) | 0.46M | 1.000 | 0.838 |
| 16 | 64 (101K) | 1.81M | 1.000 | 1.000 |
| 64 | 32 (26K) | 1.70M | 1.000 | 0.187 |
| 64 | 64 (101K) | 6.64M | 1.000 | 1.000 |

The full 12-cell grid is in the results file. The coarse question is solved by every model, including the cheapest
one (1 token, 0.08M). The reading question is at chance with 1 token for every LLM width, and partial with 4. It
needs 16 tokens, one per glyph cell, and an LLM wide enough to learn the lookup. At roughly equal inference cost
(V = 1 with width 128, V = 16 with width 64, V = 64 with width 32), only the middle point reads the board. So the
token budget should be set per task, as the post says. The narrowest LLM with 64 tokens did not learn the reading
question in 1,500 steps. We did not test whether it would with longer training.

### What freezing saves on a GPU (NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2)

One AdamW step of a 1.25B-parameter TwoTowerVLM: a 0.30B encoder on 448 px images (1,024 patches), a 2x2 merge,
a 0.87B LLM reading 256 visual and 200 text tokens, batch 8, bf16 autocast. Median of 10 steps
(`results/frozen_step_gpu.txt`):

| Setting | Trainable | Step time | Time vs all trained | FLOP model vs all trained | Peak memory |
|---|---|---|---|---|---|
| All trained | 1.254B | 232.9 ms | 1.00 | 1.00 | 48.1 GiB |
| Frozen encoder | 0.950B | 139.7 ms | 0.60 | 0.71 | 20.5 GiB |
| Frozen encoder + frozen LLM (connector trains) | 0.013B | 101.1 ms | 0.43 | 0.52 | 11.9 GiB |
| Frozen LLM, encoder trains | 0.317B | 194.0 ms | 0.83 | 0.81 | 39.5 GiB |

Freezing the LLM while the encoder trains cut the step time by 17%; the k rule predicts 19%. Freezing a tower saves
about 10 bytes of memory per frozen parameter here (the LLM row: 8.6 GiB for 0.94B parameters), which is its
weight gradient and AdamW state. At this small size activations dominate the 48 GiB, so the memory saving is modest.
For a 7B LLM the same 10 bytes per parameter is about 65 GiB. Measured step times fall faster than the FLOP model
when the encoder is frozen. The FLOP model leaves out the optimizer update and the encoder's attention, which both
shrink when the encoder stops training.

## What was not measured

- No paper-scale run. The ViT, CLIP, WebLI, native-model, PaliGemma 2 and Li et al. results in the post are quoted
  there. Here they appear only as arithmetic on their reported numbers, or as a small mechanism.
- The utility-decay and interaction tables in `04` part 1 and `05` parts 3 to 5 use illustrative constants, as the
  post's figures do. Only the exponents named as published (Hoffmann et al., Shukor et al., Li et al.) come from papers.
- The quality-quantity run uses a teacher MLP on vectors, not image-text pairs. It shows the mechanism: clean data that
  runs out and repeats, against noisy data that stays fresh.
- The glyph-board task stands in for "coarse reasoning" and "reading". It has no world knowledge, so it cannot show
  the post's finding that reasoning tasks gain from LLM size.
- No scaling law was fitted to our own runs. The fitter in `05` is checked on synthetic losses with known constants.
- The FLOP model leaves out attention scores. `01` and `02` show how large that term is.
