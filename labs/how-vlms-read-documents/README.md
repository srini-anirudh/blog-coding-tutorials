# Lab: How VLMs Read Documents

A code companion for the post [How VLMs Read Documents](https://srini-anirudh.github.io/blogs/how-vlms-read-documents/).
The post asks what a vision-language model needs in order to read a page, and what it should write down when it has.
This lab builds seven of its mechanisms from scratch in small Python files and runs each one, so you can watch the
idea happen on your own CPU.

What you can run:

- The token cost of an A4 page under native resolution, tiling and a fixed budget, checked against the real Qwen2-VL
  image processor.
- A CRNN line reader trained with CTC loss, used to measure how reading accuracy falls as text gets fewer pixels, and
  to compare a shrunk page with a full-resolution crop (the two-pass parser design).
- A LayoutLM-style encoder that finds the TOTAL on synthetic receipts, with and without 2D box embeddings.
- One invoice page written in six output formats and read back, to see what each format keeps.
- TEDS, the tree edit distance score for tables, which reproduces Figure 6 of the post exactly.
- Reading order by top-edge sort, recursive XY-cut and a small learned model, which reproduces Figure 7's scores.
- ANLS, the OmniDocBench overall formula, and olmOCR-Bench-style unit tests against edit distance.

Everything here is small and synthetic. No document model from the post is trained or evaluated, and no benchmark
number from a paper is reproduced. Where the post quotes a paper's result, this lab shows the mechanism behind it
at toy scale.

## Layout

```
labs/how-vlms-read-documents/
├── docread/
│   ├── budget.py    # tokens per page: Qwen2-VL smart_resize, InternVL tiling (reused from C7), DeepSeek-OCR modes
│   ├── ocr.py       # render text lines at a chosen size; CRNN + CTC; greedy decoding
│   ├── layout.py    # synthetic receipts; LayoutLM-style encoder with box-corner embeddings
│   ├── formats.py   # an invoice page; writers and readers for six output formats
│   ├── tables.py    # tables with spans as HTML, OTSL and Markdown, and back
│   ├── teds.py      # Zhang-Shasha tree edit distance and TEDS / TEDS-S
│   ├── order.py     # top-edge sort, XY-cut, margin rule, synthetic pages, pairwise learned orderer
│   └── metrics.py   # edit distance, ANLS, OmniDocBench overall, unit-test checks, repetition
├── scripts/         # 01 to 07, one per mechanism, plus run_all.sh
├── results/         # what the scripts printed (.txt) and the same numbers as .json
└── tests/
```

InternVL's tiling rule is not rebuilt here. `budget.py` imports `select_grid` from tutorial C7
(`c07-vlm/tinyvlm/image_processing.py`), which builds a VLM image processor step by step.

## Setup

From the repository root, after `pip install -e ".[dev]"`. Only `01_token_budget.py --check-processor` and the one
`network` test need a download (the Qwen2-VL-2B-Instruct image processor config). `04_output_formats.py` uses the
SmolLM2-135M tokenizer if it can load it and counts whitespace pieces otherwise. Set `OMP_NUM_THREADS` on a shared
machine; the numbers below used 8 threads.

## Run everything

```bash
# Tests.
python -m pytest labs/how-vlms-read-documents -m "not slow and not gpu and not network"   # 32 tests, under a minute
python -m pytest labs/how-vlms-read-documents -m network                                 # real Qwen2-VL processor
python -m pytest labs/how-vlms-read-documents -m slow                                    # short trainings

# Scripts, in post order. Or: bash labs/how-vlms-read-documents/scripts/run_all.sh
python labs/how-vlms-read-documents/scripts/01_token_budget.py --check-processor
python labs/how-vlms-read-documents/scripts/02_small_text.py          # trains the CRNN (--load reuses runs/crnn.pt)
python labs/how-vlms-read-documents/scripts/03_layout_embeddings.py   # 2 models x 3 seeds
python labs/how-vlms-read-documents/scripts/04_output_formats.py
python labs/how-vlms-read-documents/scripts/05_teds.py
python labs/how-vlms-read-documents/scripts/06_reading_order.py
python labs/how-vlms-read-documents/scripts/07_scoring.py
```

Which script covers which section of the post:

| Post section | Script | What it shows |
|---|---|---|
| Resolution and the token budget, Figure 4 | `01_token_budget.py` | Tokens per A4 page by dpi under three schemes; pixels per em |
| The pipeline tradition (CRNN, CTC); Figure 4 | `02_small_text.py` | Reading accuracy against pixels per em, and against dpi |
| Two passes: layout first, then content | `02_small_text.py` | Shrunk whole page against a native-resolution crop |
| Reading OCR words with their positions, Figure 3 | `03_layout_embeddings.py` | What box embeddings add when OCR order is lost; an OCR typo as input |
| What the model should write down, Figure 5, formats table | `04_output_formats.py` | What survives six formats; OTSL against HTML tokens |
| Tables and the TEDS metric, Figure 6 | `05_teds.py` | Figure 6 numbers; a value in the wrong column |
| Reading order and long documents, Figure 7 | `06_reading_order.py` | Figure 7 scores; four orderers on six synthetic layouts |
| How document reading is measured, Figure 8 | `07_scoring.py` | ANLS threshold; OmniDocBench overall formula; unit tests |

## Results

Every number below was measured on an AMD EPYC 7R13 with 8 threads, torch 2.15.0, CPU only. Raw output is in
`results/`, under the name of the script that wrote it. Each file starts with its hardware line and command.
The shared machine was busy, so the timings in the result files are rough. Accuracies do not depend on load.

### Tokens per A4 page (`01_token_budget.py`)

| dpi | Post formula | Native resolution (Qwen2-VL rule) | Real Qwen2-VL processor | 448-px tiles, Figure 4 count |
|---|---|---|---|---|
| 72 | 639 | 630 | 630 | 1,280 (4 tiles + thumbnail) |
| 150 | 2,775 | 2,772 | 2,772 | 3,328 (12 tiles + thumbnail) |
| 200 | 4,932 | 4,956 | 4,956 | 6,400 (24 tiles + thumbnail) |
| 300 | 11,098 | 11,125 | 11,125 | 10,496 (40 tiles, the cap, + thumbnail) |

The post's formula is within 1% of the real processor. InternVL's own rule picks the grid by aspect ratio, so an A4 page
always gets 35 tiles with a cap of 40 (9,216 tokens with a thumbnail) and 6 tiles with a cap of 12 (1,792 tokens),
whatever the dpi. DeepSeek-OCR's four modes cost 64, 100, 256 and 400 tokens. They shrink the page to 44, 55, 88 and 110
dpi, which leaves a 10-point letter 6.1, 7.6, 12.2 and 15.2 pixels per em.

### Small text with a CRNN reader (`02_small_text.py`)

One CRNN (653K parameters), trained 3,000 steps on lines at 3 to 28 pixels per em, tested on 400 held-out lines per row.
Lines are amounts like `29,968.56`, words and codes like `ZTF-5200`.

| Pixels per em | Line exactly right | Amounts exactly right |
|---|---|---|
| 3 | 0.34 | 0.81 |
| 4 | 0.42 | 0.67 |
| 5 | 0.95 | 1.00 |
| 6 | 0.74 | 0.83 |
| 8 | 0.97 | 1.00 |
| 12 | 1.00 | 1.00 |

The curve is not smooth. Pixels per em 5 beats 6 because the font renders some sizes more cleanly than others, and the
reader had few training lines at each exact size. Read the trend (poor below 5, near perfect from 8), not single rows.
Real page text at the post's resolutions:

| dpi | 10-pt text, line exact | 6.5-pt footnote, line exact |
|---|---|---|
| 72 | 0.995 | 0.915 |
| 110 | 0.995 | 0.973 |
| 150 | 1.000 | 0.998 |
| 200 | 1.000 | 0.993 |

The same reader on a page rendered at 200 dpi, then shrunk so the whole page fits a square input, against a crop of
each line at full resolution:

| Input | 10-pt text exact | 6.5-pt footnote exact |
|---|---|---|
| Native crop of the line (second pass) | 0.998 | 1.000 |
| Whole page shrunk to 1280 px | 0.995 | 0.960 |
| Whole page shrunk to 1036 px | 0.993 | 0.920 |
| Whole page shrunk to 1024 px | 0.993 | 0.900 |
| Whole page shrunk to 640 px | 0.912 | 0.487 |

This shows the direction of the post's argument, not its size. A single-line reader with a clean font is far more
forgiving than a model reading a dense, noisy page: this reader was trained on shrunk lines, so it holds up until
about 5 pixels per em. The footnote falls first, and a 640-px page loses half of its footnotes. Training took 1,868 s
on a heavily shared machine, which is more than the usual time for this script.

### Box embeddings on synthetic receipts (`03_layout_embeddings.py`)

Task: pick the amount that belongs to TOTAL. Every amount is the same `[NUM]` token, so the model must use position.
Chance is 0.115. 6,000 training receipts, 1,000 test receipts per order, 3,000 steps, mean of 3 seeds.

| OCR word order at test time | Text + 1D position | Text + 1D + box embeddings |
|---|---|---|
| Line by line (rows) | 0.994 | 0.982 |
| Left half, then right half (columns) | 0.828 | 0.977 |
| Shuffled | 0.108 | 0.961 |
| Line by line, "TOTAL" misread as "T0TAL" | 0.261 | 0.311 |

With boxes, the order the OCR engine returns stops mattering. The last row is the post's catch: once the key word is
misread, both models fall to about a quarter of their accuracy. The seeds differ a lot there (0.20 to 0.43 for the box
model), so the two numbers are not distinguishable. The input error is not repaired.

### Output formats (`04_output_formats.py`)

One invoice page with 8 elements, 32 words, a table with a merged header cell and 3 styled elements. Tokens use the
SmolLM2 tokenizer, with each DocTags tag counted as one token.

| Format | Tokens | Words kept | Headings | Table TEDS | Box IoU | Styles kept |
|---|---|---|---|---|---|---|
| Plain text | 113 | 1.00 | 0 / 2 | 0.88 | 0.00 | 0 / 3 |
| Markdown | 143 | 1.00 | 2 / 2 | 0.88 | 0.00 | 2 / 3 |
| HTML | 245 | 1.00 | 2 / 2 | 1.00 | 0.00 | 3 / 3 |
| JSON fields | 63 | 0.16 | 0 / 2 | 0.00 | 0.00 | 0 / 3 |
| DocTags | 185 | 1.00 | 2 / 2 | 1.00 | 0.98 | 0 / 3 |
| Layout HTML | 524 | 1.00 | 2 / 2 | 1.00 | 1.00 | 3 / 3 |

The table alone costs 112 tokens as HTML, 63 as OTSL and 65 as Markdown. OTSL and HTML read back to TEDS 1.00.
Markdown reads back to 0.88 because it cannot write the span. The plain-text reader has to guess which lines are the
table, which gives it the same 0.88 here; that number depends on how the guess is coded.

### TEDS (`05_teds.py`)

| Case | TEDS | TEDS-S | Flattened text |
|---|---|---|---|
| Figure 6 prediction (cost 2.25 over 13 nodes) | 0.827 | 0.846 | 0.971 |
| One value in the wrong column | 0.882 | 1.000 | 1.000 |
| Last row missing | 0.765 | 0.765 | 0.841 |
| Figure 6 ground truth through Markdown | 0.846 | 0.846 | 1.000 |

These match the post's 0.83, 0.85 and 0.97. A value moved to the wrong column leaves the flattened text identical, so a
text score gives it full marks.

### Reading order (`06_reading_order.py`)

On the Figure 7 page: top to bottom 0.62, column-aware XY-cut 0.15, XY-cut after dropping the header and page number 0.00.
On 300 synthetic pages per layout (mean edit distance to the reference order, lower is better):

| Layout | Top to bottom | XY-cut | XY-cut without margin blocks | Learned pairwise model |
|---|---|---|---|---|
| 1 column | 0.210 | 0.210 | 0.000 | 0.000 |
| 2 columns | 0.762 | 0.127 | 0.006 | 0.061 |
| 3 columns | 0.852 | 0.088 | 0.001 | 0.121 |
| 2 columns + full-width figure | 0.511 | 0.154 | 0.001 | 0.318 |
| 3 columns + figure over 2 | 0.808 | 0.131 | 0.034 | 0.174 |
| 2 columns, boxes cross the gutter | 0.763 | 0.471 | 0.401 | 0.094 |

A rule works until the boxes are loose; the learned model is worse on clean pages and better on loose ones. The
learned model sees two boxes at a time, so it cannot learn page-wide rules such as a full-width figure splitting two
column bands.

### Scoring (`07_scoring.py`)

ANLS gives `1,580.00` for gold `1,500.00` a score of 0.875, and `15` a score of 0. The OmniDocBench overall formula
reproduces all 15 overall scores in Figure 8 from their components, with a largest gap of 0.02 points. On a small
page, moving a caption costs edit distance 0.436 and passes every unit test, while a one-digit error costs 0.009 and
fails the unit test that names it.

### Tests

On the same hardware: 32 offline tests pass (`not slow and not gpu and not network`). The 1 `network` test passes
(real Qwen2-VL image processor). The 2 `slow` tests pass in about 9 minutes on a busy machine. See `results/tests.txt`.

## What was not measured

- No real document model from the post (Donut, Nougat, GOT, olmOCR, MinerU2.5, PaddleOCR-VL, Qwen2.5-VL and the
  others) was run. The paper and leaderboard numbers in the post are not reproduced here. `07_scoring.py` only checks
  that the OmniDocBench overall formula gives back the overall scores the post quotes from their components.
- No real benchmark data (DocVQA, FUNSD, CORD, PubTabNet, OmniDocBench) is used. Receipts, pages, tables and text
  lines are synthetic, and the CRNN reads single lines in one font (Pillow's bundled Aileron).
- TextMonkey's token-redundancy measurement, DeepSeek-OCR's compression ratios, window against full attention in the
  encoder, multi-page reading, CDM for formulas and mask-based grounding are discussed in the post but not built here.
- The TEDS tree is table, rows, cells, as Figure 6 draws it. PubTabNet's reference code also keeps thead and tbody
  nodes, so its numbers on the same tables can differ slightly.
- No GPU was used. Nothing in this lab needs one.
