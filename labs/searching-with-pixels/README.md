# Lab: Searching With Pixels

The code companion to the post [Searching With Pixels: Multimodal Retrieval and Visual RAG](https://srini-anirudh.github.io/blogs/searching-with-pixels/).
Part of [blog-coding-tutorials](https://github.com/srini-anirudh/blog-coding-tutorials).

The post explains how to find the right page among many and hand it to a vision-language model. This lab builds the
search half small enough to run on a CPU. Every model here is trained from scratch in a minute or two, on synthetic
"pages" where we know the right answer exactly. One section loads a real CLIP model.

The task is built around the post's bar-chart example, where "2023" and "Asia" must be found in two different places
on one page. Each page is a 32 x 32 image cut into 16 patches. Six patches hold a glyph: one of 16 shapes in one of 8
colours. A query names two glyphs, such as "cyan shape12 magenta shape11". Every page has a **twin** with the same six
shapes and the same six colours, but with the colours on different shapes. A model that only knows *which* shapes and
colours are on a page cannot tell a page from its twin. It has to know which colour sits on which shape.

## What you build

| Module | What it does |
|---|---|
| `pixsearch/pages.py` | The synthetic pages, their twins and clones, binding queries, and exact relevance labels. |
| `pixsearch/models.py` | A patch encoder and a query encoder (small Transformers), and the three scorers: `BiEncoder` (one vector, cosine), `LateInteraction` (one vector per patch, `maxsim`), `CrossEncoder` (query and patches in one sequence). |
| `pixsearch/train.py` | Symmetric InfoNCE with in-batch negatives, false-negative masking, optional mined negatives; listwise training for the cross-encoder; the closed-form InfoNCE gradient. |
| `pixsearch/metrics.py` | Recall@k, Precision@1, MRR, DCG and nDCG@k (graded or binary). |
| `pixsearch/retrieve.py` | Index a corpus once, score queries against it, grade, and time. |
| `pixsearch/index.py` | The storage formula; merging and pruning patch vectors; fp16, int8 and binary quantisation; binary search with full-precision rescoring. |
| `pixsearch/gap.py` | Modality-gap statistics, the shared-offset argument, and wrong-modality rates in mixed pools. |

## Setup

From the repository root, with the repo's environment (`pip install -e ".[dev]"`). No extra packages. Script 05
downloads [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32), about 600 MB; pass
`--skip-clip` to stay offline. The models are tiny, so the scripts use one CPU thread by default (set
`PIXSEARCH_THREADS` to change it). On our shared machine, more threads made them slower.

## Run everything

From the repository root:

```bash
bash labs/searching-with-pixels/scripts/run_all.sh        # set PY to choose the Python; about 29 minutes here
```

| Command | What it shows | Output |
|---|---|---|
| `python labs/searching-with-pixels/scripts/01_metrics.py` | Recall@k, MRR and nDCG@5 for one relevant page at ranks 1 to 10, and graded nDCG. | `results/01_metrics.txt` |
| `python labs/searching-with-pixels/scripts/02_three_scorers.py` | Trains bi16, bi64, bi256, late16 and a cross-encoder (1,500 steps each), grades them on 2,000 held-out pages, times them, prints a MaxSim map. Checkpoints go to `runs/` (not in git). | `results/02_three_scorers.{txt,json}` |
| `python labs/searching-with-pixels/scripts/02b_bi_longer.py` | bi256 trained 6,000 steps: is the bi-encoder's failure a training limit? | `results/02b_bi_longer.{txt,json}` |
| `python labs/searching-with-pixels/scripts/03_cascade.py` | bi64 retrieves the top k; late16 or the cross-encoder reranks. Plus the post's Figure 8 token arithmetic. | `results/03_cascade.{txt,json}` |
| `python labs/searching-with-pixels/scripts/04_negatives.py` | The InfoNCE gradient against autograd, how little weight easy negatives get, and four negative strategies including false negatives. | `results/04_negatives.{txt,json}` |
| `python labs/searching-with-pixels/scripts/05_modality_gap.py [--skip-clip]` | The gap in our own towers, then in CLIP ViT-B/32 on drawn pictures, and modality bias in a mixed pool. | `results/05_modality_gap.{txt,json}` |
| `python labs/searching-with-pixels/scripts/06_index_cost.py` | The storage formula with ColPali's numbers, then merging, pruning and quantising the late16 index. | `results/06_index_cost.{txt,json}` |

Scripts 03 to 06 load the checkpoints that script 02 writes. Each result file starts with the hardware and the command.

## Tests

```bash
python -m pytest labs/searching-with-pixels -m "not slow and not gpu and not network"   # 29 tests, about 10 s
python -m pytest labs/searching-with-pixels -m "slow and not network"                   # every script at toy size
python -m pytest labs/searching-with-pixels -m "network"                                # CLIP part of script 05 at toy size
```

The offline tests check the metrics against the post's worked examples and a hand-computed graded nDCG, the twin and
clone construction, exact relevance labels, MaxSim against a Python loop, that page vectors never depend on the
query or on other pages, the closed-form InfoNCE gradient against autograd, false-negative masking, ColPali's 263,680
bytes per page, merging and quantisation, and the shared-offset argument. The smoke tests run with
`PIXSEARCH_FAST=1`, which writes to `runs/fast/` and never touches `results/`.

## Results

All numbers were measured on an **AMD EPYC 7R13 CPU, 1 thread**, float32, torch 2.15, on a shared machine that was
heavily loaded by other jobs. Timings are wall-clock and only comparable within one table. Every model is trained from
scratch on synthetic pages, so none of these numbers can be compared with the paper results the post quotes.

### Three ways to score a match (`02_three_scorers`, `02b_bi_longer`)

Held-out corpus: 2,000 pages (1,000 page and twin pairs), 500 queries, 4.91 relevant pages per query on average
(two glyphs often appear together by chance). "Beats twin" is the share of queries whose source page outscores its
twin; 50 is chance. The cross-encoder must run once per page, so it was scored on the first 100 queries only.

| Model | Stored floats per page | Encoder passes per query | ms per query | P@1 | nDCG@5 | R@10 | Beats twin |
|---|---|---|---|---|---|---|---|
| Bi-encoder, 16-d | 16 | 1 | 0.04 | 8.2 | 8.3 | 50.4 | 53.8 |
| Bi-encoder, 64-d | 64 | 1 | 0.05 | 11.0 | 12.0 | 66.0 | 50.4 |
| Bi-encoder, 256-d | 256 | 1 | 0.07 | 11.8 | 12.6 | 66.8 | 50.4 |
| Bi-encoder, 256-d, 6,000 steps | 256 | 1 | | 13.0 | 16.2 | 73.0 | 51.8 |
| Late interaction, 16 x 16-d | 256 | 1 | 0.92 | **100.0** | **100.0** | 100.0 | **100.0** |
| Cross-encoder | 0 (raw page) | 2,000 | 271 | 57.0 | 67.9 | 100.0 | 98.0 |

Late interaction and the 256-d bi-encoder store the same 256 floats per page. Only the one that keeps a vector per
patch can tell a page from its twin. The bi-encoders sit at chance on twins at every size, and four times more
training did not change that. A 256-d vector has room for all 128 glyphs, so this is not proven to be impossible; it
is what mean pooling learned here. The MaxSim map for "yellow shape3 white shape2" shows the mechanism: "yellow" and
"shape3" both pick the cell holding the yellow shape 3 (similarity 0.87 and 0.77), and "white" and "shape2" both pick
the white shape 2. Late interaction pays at query time (0.92 ms against 0.05 ms per query over 2,000 pages, on this
CPU). The cross-encoder pays far more: 271 ms per query, one forward pass per page. It was trained to choose among 8
pages (the page, its twin and 6 others), and as a full-corpus scorer it beats the twin 98% of the time but puts a
non-relevant page first in 43% of queries.

Training time per model was about 100 s for each dual model and 462 s for the cross-encoder.

### Cascades (`03_cascade`)

bi64 retrieves the top k of 2,000 pages. Then late16 or the cross-encoder reorders those k.

| k | Stage-1 Recall@k | late16 rerank P@1 | nDCG@5 | cross rerank P@1 | nDCG@5 | Cross ms per query |
|---|---|---|---|---|---|---|
| 1 | 11.0 | 11.0 | 4.1 | 11.0 | 4.1 | 0.7 |
| 5 | 42.4 | 42.4 | 18.8 | 36.8 | 17.8 | 1.3 |
| 10 | 66.0 | 66.0 | 34.0 | 49.4 | 30.8 | 1.4 |
| 20 | 85.6 | 85.6 | 53.7 | 56.0 | 45.5 | 2.7 |
| 50 | 96.8 | 96.8 | 85.0 | 57.0 | 62.4 | 5.5 |
| 100 | 99.8 | 99.8 | 98.9 | 57.6 | 66.6 | 13.0 |

The reranker can never beat the first stage's recall: late16's P@1 equals stage-1 Recall@k in every row. The first
stage is weak (P@1 11.0) but its Recall@100 is 99.8, so reranking 100 of 2,000 pages recovers nearly everything.
Reranking 100 pages with the cross-encoder took 13 ms per query; scoring all 2,000 took 271 ms in script 02 (same CPU, a different run). The token arithmetic
of the post's Figure 8 is in the result file (rerank 50 pages: 51,500 tokens; read every one of 40,000 pages: 40
million). It is arithmetic, not a measurement.

### Negatives (`04_negatives`)

The closed-form InfoNCE gradient matches autograd (max difference 3.8e-06). On the trained late16 model over 2,000
pages, the hardest negative gets 18.5% of the push-away weight and the top 10 get 68.9%. 84.9% of negatives get a
weight below 1e-6, so they give almost no gradient.

Part B trains late16 for 1,000 steps on one fixed pool of 4,096 pages, four ways. A pool query has 9.4 relevant
pages on average, and 26% of queries also have an exact clone of their page.

| Negatives | P@1 | nDCG@5 | Beats twin | Mined negatives that were really relevant |
|---|---|---|---|---|
| Random batches | 62.2 | 69.0 | 95.2 | |
| Twins in each batch | 57.4 | 65.4 | 96.2 | |
| Mined from the model's own index, filtered with the labels | **100.0** | **100.0** | 100.0 | 0% |
| Mined, only the labelled page skipped | 71.8 | 74.7 | 97.4 | rises from 0.1% to 78% over training |

Mining the hardest non-relevant pages with the model being trained (ANCE-style) lifts P@1 from 62 to 100. Without
filtering, the miner soon picks pages that really answer the query: by the last refresh, 78% of mined "negatives"
were relevant, and the gain mostly disappears. Twins in the batch did not help in this single-seed run. Random batches already learned to bind
(95% beats twin), so the remaining errors came from other near-matches that only the model's own index finds.

### The modality gap (`05_modality_gap`)

Our own two towers at random initialisation: page vectors have a mean pairwise cosine of 0.977 (a narrow cone), query
vectors 0.410, gap 1.10 between the centroids. After training (bi64, temperature 0.06), the gap fell to 0.18 and the
cones opened (0.023 and 0.046). So on this toy task, contrastive training closed most of the gap it started with.
Subtracting a shared offset from every page vector moved each query's 2,000 dot-product scores by one constant
(largest deviation 4e-07) and left every top-1 unchanged, as the post's derivation says.

CLIP ViT-B/32 on 48 drawn pictures (6 shapes x 8 colours) with captions "a photo of a {colour} {shape}.". The mixed
pool adds 47 captions of *other* pictures in another template, so every text candidate is a wrong answer.

| Vectors | Gap | Image-only pool top-1 | Mixed pool top-1 | Text candidate ranked first |
|---|---|---|---|---|
| Raw CLIP | 1.026 | 75.0% | 0.0% | 100.0% |
| Each modality's mean removed | 0.366 | 85.4% | 58.3% | 41.7% |

Matched image-caption pairs have a mean cosine of 0.331, but a query and a wrong caption have 0.686. So in the mixed
pool a caption of the wrong picture beat the right picture for every query. Removing each modality's mean (computed on
this pool) helped but did not fix it. These are 48 drawn pictures, not Flickr30K, COCO or M-BEIR.

### Index cost (`06_index_cost`)

The storage formula reproduces the post's numbers: 1,030 x 128 x 2 bytes = 263,680 bytes (257.5 KiB) per ColPali
page, 64.4 times one 2,048-d fp16 vector, 263.7 GB for a million pages. The late16 index (16 vectors of 16 dims per
page, nDCG@5 100.0 in fp32):

| Variant | Vectors per page | Bytes per page | nDCG@5 | P@1 |
|---|---|---|---|---|
| fp32 | 16 | 1,024 | 100.0 | 100.0 |
| fp16 / int8 | 16 | 512 / 256 | 100.0 / 100.0 | 100.0 / 100.0 |
| Binary (sign only) | 16 | 32 | 69.8 | 71.4 |
| Binary, rescore top 20 / 50 in fp32 | 16 | 32 + short list | 95.0 / 98.6 | 98.2 / 99.4 |
| Merge, factor 2 / 3 / 4 | 8 / 6 / 4 | 512 / 384 / 256 | 100.0 / 90.0 / 50.0 | 100.0 / 96.8 / 67.0 |
| Prune by norm, keep 8 / 6 / 4 | 8 / 6 / 4 | 512 / 384 / 256 | 68.8 / 56.1 / 34.9 | 82.6 / 71.6 / 46.4 |
| Prune at random, keep 8 / 6 / 4 | 8 / 6 / 4 | 512 / 384 / 256 | 39.0 / 24.3 / 16.8 | 70.4 / 47.2 / 28.2 |

At every budget, merging kept more quality than pruning, which matches the direction of the Light-ColPali finding the
post cites. Pruning is query-agnostic: it must guess which patch a future query will need. Our pages have 10 blank
patches out of 16, which likely explains why merging by factor 2 loses nothing. Real pages have far fewer empty patches, and the paper's
retention numbers were not reproduced.

## Post sections and scripts

| Post section | Script |
|---|---|
| Why a VLM needs a search engine | `03_cascade.py` (token arithmetic) |
| Three ways to score a match; Cascades | `02_three_scorers.py`, `02b_bi_longer.py`, `03_cascade.py` |
| How retrieval is measured | `01_metrics.py`, `pixsearch/metrics.py` |
| Contrastive dual encoders as retrievers | `pixsearch/train.py`, `02_three_scorers.py` |
| The modality gap and mixed-modality search | `05_modality_gap.py` |
| Retrieving document pages without OCR (late interaction over patches, Figure 5) | `02_three_scorers.py` (MaxSim map) |
| What multi-vector indexes cost, and how to shrink them | `06_index_cost.py` |
| Reranking with a VLM | `03_cascade.py` (a toy cross-encoder, not a VLM) |
| Training data, hard negatives and synthetic queries | `04_negatives.py` |

## What was not measured

- No real documents, no OCR pipeline, and no ColPali, DSE, VisRAG or VLM embedder. The pages are synthetic glyph
  grids. None of the post's ViDoRe, MMEB, M-BEIR or MRAG-Bench numbers were reproduced.
- No VLM reranker (MonoQwen2-VL style yes/no scoring). The cross-encoder is a 2-layer Transformer.
- The generation half of visual RAG (feeding pages to a VLM, citations, verification) and video retrieval are not covered.
- No approximate nearest-neighbour index, PLAID, MUVERA or Matryoshka truncation. Every search is exhaustive.
- The UniIR wrong-modality rates, the jina-embeddings-v4 pair cosines and the E5-V prompt trick were not tested; the CLIP
  run uses 48 drawn pictures only.
- Nothing here needs a GPU, and no GPU was used.
