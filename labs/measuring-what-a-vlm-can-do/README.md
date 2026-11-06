# Lab: Measuring What a VLM Can Do

Code companion for [Measuring What a VLM Can Do](https://srini-anirudh.github.io/blogs/measuring-what-a-vlm-can-do/).

The post says a benchmark number is a property of the whole pipeline: task, data, metric and protocol. This lab
builds that pipeline in small Python and runs it. It has two halves.

The first half builds the scoring rules and checks them on the post's worked examples. It covers VQA accuracy with
its ten leave-one-out subsets, ANLS, relaxed accuracy, IoU and Acc@0.5, gIoU against cIoU, the POPE statistics,
CHAIR and Elo. Simulations then show CircularEval, the noise floor of a small benchmark, and an arena fitted with
Bradley–Terry.

The second half runs two real public models on two real benchmarks. The models are SmolVLM-256M-Instruct and
SmolVLM-500M-Instruct. The benchmarks are MMStar (1,500 questions) and POPE (3 × 3,000 questions). The same saved
replies are then scored under different protocols. That shows the post's protocol claims on real outputs: the parser
changes the score, the prompt changes the score, tiling changes the input, the blind test splits the score into
seeing and guessing, and CircularEval removes position bias.

These are small models, and none of the post's large-model numbers are reproduced here. MMStar's own numbers for
GPT-4V or Gemini, BLINK, MMMU and the rest come from the papers.

C8 already builds Levenshtein distance, ANLS, relaxed accuracy, box IoU and the click test in
`c08-vlm-training/vlmtrain/metrics.py`. This lab imports them instead of rebuilding them.

## Layout

```
labs/measuring-what-a-vlm-can-do/
├── benchlab/
│   ├── metrics.py   # VQA normaliser + accuracy, Acc@0.5, gIoU / cIoU, POPE, CHAIR (ANLS, IoU from C8)
│   ├── mc.py        # MMStar option parsing, prompts, three answer extractors, CircularEval rotations
│   ├── stats.py     # standard error, bootstrap and paired bootstrap, MG / ML, Elo, Bradley–Terry fit
│   ├── runner.py    # MMStar and POPE loaders, batched SmolVLM and SmolLM2 generation
│   └── report.py    # writes results/<script>.txt with a hardware line and the command
├── scripts/
│   ├── 01_metrics.py        # every worked metric example in the post
│   ├── 02_circular_eval.py  # CircularEval on simulated guessers and position-biased models
│   ├── 03_noise_floor.py    # SE = sqrt(p(1-p)/n), checked by simulation
│   ├── 04_arena.py          # simulated votes -> Bradley–Terry ratings, and a style effect
│   ├── 10_run_models.py     # real models on MMStar or POPE; saves every raw reply (GPU)
│   └── 11_analyse.py        # scores the saved replies under each protocol (CPU, seconds)
├── results/        # outputs written by the scripts; raw/ holds the model replies
└── tests/          # 28 offline tests, 4 network tests (2 of them slow)
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. The CPU scripts need nothing else. The model runs
download about 2 GB of weights and 300 MB of data from the Hugging Face Hub: `HuggingFaceTB/SmolVLM-256M-Instruct`,
`HuggingFaceTB/SmolVLM-500M-Instruct`, `HuggingFaceTB/SmolLM2-135M-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct`,
the dataset `Lin-Chen/MMStar` and the dataset `lmms-lab/POPE`.

## Run everything

```bash
python -m pytest labs/measuring-what-a-vlm-can-do -m "not slow and not gpu and not network"   # 28 tests, seconds
python -m pytest labs/measuring-what-a-vlm-can-do -m network                                 # 4 tests, downloads (2 are slow)

# CPU, seconds each.
python labs/measuring-what-a-vlm-can-do/scripts/01_metrics.py
python labs/measuring-what-a-vlm-can-do/scripts/02_circular_eval.py
python labs/measuring-what-a-vlm-can-do/scripts/03_noise_floor.py
python labs/measuring-what-a-vlm-can-do/scripts/04_arena.py

# Real models. One GPU, 8 CPU workers for image tiling; 7 to 11 minutes per run on an H100 (times below).
# On CPU, add --limit 20 and drop --workers.
python labs/measuring-what-a-vlm-can-do/scripts/10_run_models.py --model 256m --bench mmstar --batch-size 32 --workers 8
python labs/measuring-what-a-vlm-can-do/scripts/10_run_models.py --model 500m --bench mmstar --batch-size 32 --workers 8
python labs/measuring-what-a-vlm-can-do/scripts/10_run_models.py --model 256m --bench pope --batch-size 32 --workers 8
python labs/measuring-what-a-vlm-can-do/scripts/10_run_models.py --model 500m --bench pope --batch-size 32 --workers 8
python labs/measuring-what-a-vlm-can-do/scripts/11_analyse.py
```

`10_run_models.py` only generates text. All scoring happens in `11_analyse.py`, so you can add an extractor or a
metric and re-score without a GPU.

## What the model runs measure

Every MMStar condition uses greedy decoding with at most 24 new tokens.

| Condition | Image | Prompt | Used for |
|---|---|---|---|
| `fmt` | yes, tiling on | MMStar layout + "Answer with the option's letter from the given choices directly." | the headline score |
| `bare` | yes, tiling on | MMStar layout, no format hint | the prompt is part of the protocol |
| `fmt_notile` | yes, one 512 px view | as `fmt` | input settings are part of the protocol |
| `blind` | none | as `fmt` | S_wv in MMStar's multi-modal gain |
| `llm` | none, SmolLM2 Instruct | as `fmt` | S_t in MMStar's multi-modal leakage |
| `rot1` to `rot3` | yes, tiling on | as `fmt`, options rotated by 1, 2, 3 | CircularEval |

Three extractors score each reply. `strict` accepts only a bare letter such as `B` or `(B).`. `permissive` accepts
exactly one standalone option letter anywhere in the reply. `content` tries `permissive`, then matches the text of
exactly one option. None of them calls a language model, unlike VLMEvalKit's and MathVista's last-resort LLM
extractor. A reply no extractor can read scores 0.

The base LLMs are SmolLM2-135M-Instruct and SmolLM2-360M-Instruct. Their shapes match the decoders inside the two
SmolVLMs (576 and 960 hidden units, 30 and 32 layers). They are the public text checkpoints, not the decoder weights
after multimodal training. This is the same comparison MMStar makes: the VLM against the LLM it was built from.

## Results

Every number names its hardware. The simulation and scoring scripts ran on AMD EPYC 7R13, 8 threads, torch 2.15.0.
The model replies were generated on NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2, bf16, greedy decoding, batch
size 32, with 8 CPU worker processes for image resizing and tiling. Raw replies are in `results/raw/`. Scores are in
`results/11_analyse.txt` and `results/11_analyse.json`. Each model was run once; greedy decoding is deterministic up to
GPU numerics, and no seeds vary.

Tests on AMD EPYC 7R13, 8 threads: 28 offline tests pass (4 deselected). With `network` and `slow` included, all 32
pass (about 35 s with the models cached).

### Metrics and simulations (CPU, seconds)

| Check | Result | File |
|---|---|---|
| VQA accuracy, ten humans (4 red, 3 pink, 2 maroon, 1 orange): red / pink / maroon / orange / blue | 1.00 / 0.90 / 0.60 / 0.30 / 0.00 | `01_metrics.txt` |
| Same, with the naive `min(n/3, 1)` rule | 1.00 / 1.00 / 0.67 / 0.33 / 0.00 | `01_metrics.txt` |
| "Two people." against ten answers "2" | 0.0 (the normaliser does not repair a sentence) | `01_metrics.txt` |
| ANLS, gold SPRINGFIELD: `SPRlNGFIELD` / `SPRING` / `SHELBYVILLE` | 0.909 / 0.545 / 0 | `01_metrics.txt` |
| ANLS, `7,250` against gold `1,250` | 0.80 | `01_metrics.txt` |
| Relaxed accuracy, 2020 against 2019 | correct | `01_metrics.txt` |
| Same 18 px shift: IoU on a 300 px box / on a 30 px box | 0.887 pass / 0.250 fail | `01_metrics.txt` |
| gIoU / cIoU for one mask at IoU 0.9 and one at 0.2 | 0.55 / 0.893 | `01_metrics.txt` |
| POPE statistics on a balanced 10: calibrated / yes-biased accuracy and yes-ratio | 0.80, 0.50 / 0.70, 0.80 | `01_metrics.txt` |
| Elo, 400-point gap | 0.909 = 10/11 | `01_metrics.txt` |
| CircularEval, 1,500 simulated 4-option questions: uniform guesser / always "A", single pass | 0.224 / 0.249 | `02_circular_eval.txt` |
| Same two under CircularEval | 0.003 / 0.000 | `02_circular_eval.txt` |
| Simulated model that knows 70%, guesses the rest: single pass / CircularEval | 0.787 / 0.707 | `02_circular_eval.txt` |
| Standard error at p = 0.6: n = 1,000 / 218 / 60 (formula; simulation agrees to 0.1 pt) | 1.5 / 3.3 / 6.3 points | `03_noise_floor.txt` |
| Two models at 0.60 and 0.62 true accuracy: chance the worse one scores at least as high, n = 60 / 1,000 / 5,000 | 0.444 / 0.183 / 0.020 | `03_noise_floor.txt` |
| Arena, five models, 50,000 votes: largest rating error | 2.6 Elo points | `04_arena.txt` |
| Arena with a style bonus for one model that is truly rated 4th: its fitted rank at bonus 0 / 0.5 / 1.0 | 4 / 3 / 2 | `04_arena.txt` |

### Real models on MMStar, 1,500 questions (accuracy %, NVIDIA H100 80GB HBM3)

Chance is 25% on four options. Always answering "A", "B", "C" or "D" scores 28.6, 29.8, 20.6 or 21.0.

| Condition | SmolVLM-256M `strict` / `permissive` / `content` | SmolVLM-500M `strict` / `permissive` / `content` |
|---|---|---|
| Image, format hint (headline) | 5.1 / 23.7 / 25.9 | 4.5 / 35.6 / 35.7 |
| Image, no format hint | 0.8 / 18.1 / 21.2 | 6.7 / 27.5 / 30.9 |
| Image, hint, tiling off (64 image tokens, not about 877) | 2.0 / 21.8 / 22.9 | 0.2 / 34.3 / 34.3 |
| No image | 0.0 / 17.7 / 19.5 | 0.0 / 27.8 / 27.8 |
| Text-only base LLM (SmolLM2-135M / -360M Instruct) | 0.7 / 22.1 / 24.3 | 3.5 / 26.1 / 26.2 |

| Quantity | SmolVLM-256M | SmolVLM-500M |
|---|---|---|
| Headline (`content` extractor), bootstrap 95% interval | 25.9 [23.8, 28.3] | 35.7 [33.3, 38.2] |
| Formula standard error at that accuracy | 1.1 points | 1.2 points |
| Multi-modal gain MG = S_v − S_wv | 6.5 points (image minus no image, paired interval [+3.9, +9.2]) | 7.9 points (paired interval [+5.2, +10.7]) |
| Multi-modal leakage ML = max(0, S_wv − S_t) | 0.0 | 1.6 points |
| Single pass / mean over four rotations / CircularEval | 25.9 / 25.9 / 0.9 | 35.8 / 34.1 / 7.3 |
| Letter picked: A / B / C / D (unparsed) | 238 / 135 / 286 / 768 (73) | 465 / 518 / 244 / 265 (8) |
| GPU run time for all conditions, including CPU image tiling | 642 s | 569 s |

Per category, 500M with and without the image: coarse perception 50.8 against 28.4, math 34.4 against 25.2, and
fine-grained perception 28.4 against 34.4 (the image did not help there). The 500M model beats the 256M model by 9.8
points, paired bootstrap 95% interval [+6.7, +13.0].

What these runs show:

- The extractor dominates. Both models usually reply `D: The suitcase is beneath the book.` A strict parser accepts
  none of those, so the same replies score 5.1 or 4.5 under `strict` and 25.9 or 35.7 under `content`.
- The prompt is part of the metric. Dropping the format hint costs 4.7 points (256M) and 4.8 points (500M).
- Tiling is an input setting. With one 512 px view instead of about 877 image tokens, 500M loses 1.4 points (inside
  the headline's 95% interval) and 256M loses 3.0 points (just outside it). That interval is for one score, not for a
  paired difference, and the lab did not run the paired test. So this run suggests, but does not show, that tiling
  matters for MMStar. It shows the setting must be stated.
- The blind test works. Without the image, 500M falls from 35.7 to 27.8, still above the 25% chance level and above
  the 26.2 of its base LLM. The 256M model's blind score (19.5) is below chance and below its base LLM (24.3), so its
  leakage is zero.
- The 256M model has a position bias. It picks "D" in 768 of 1,500 replies (51%), where D is the gold letter 21% of the
  time. CircularEval cuts its score from 25.9 to 0.9. The 500M model keeps 7.3.
- Chance level is not 25% for a model that always answers one letter. "B" alone scores 29.8, which is above the 256M
  model's headline. At this model size the 256M number is not distinguishable from a position habit.

### Real models on POPE, 3 × 3,000 yes/no questions (NVIDIA H100 80GB HBM3)

| Model, setting | Accuracy | Precision | Recall | F1 | Yes-ratio |
|---|---|---|---|---|---|
| 256M, random | 85.5 | 97.1 | 73.2 | 83.5 | 0.377 |
| 256M, popular | 83.6 | 92.4 | 73.3 | 81.7 | 0.396 |
| 256M, adversarial | 80.9 | 86.9 | 72.7 | 79.2 | 0.418 |
| 500M, random | 88.6 | 96.5 | 80.2 | 87.6 | 0.416 |
| 500M, popular | 85.7 | 90.1 | 80.3 | 84.9 | 0.446 |
| 500M, adversarial | 83.2 | 85.2 | 80.4 | 82.7 | 0.472 |

The settings order the way the post says: accuracy falls from random to popular to adversarial for both models (by
4.6 and 5.4 points). Precision falls fastest, because the adversarial setting asks about objects that co-occur with
what is in the image. The yes-ratio is below 0.5 for both models, so these two models under-claim objects instead of
inventing them. The post's rule that a yes-ratio far above 0.5 signals hallucination is not triggered here. All 18,000
replies parsed. POPE's prompt in this lab ends with "Answer the question using a single word or phrase."

## Which script covers which section of the post

| Post section | Script or file |
|---|---|
| What a benchmark score claims; the anatomy of a benchmark (Figure 1) | `10_run_models.py` conditions `fmt`, `bare`, `fmt_notile`; `11_analyse.py` |
| Answers have many valid surface forms; exact match | `benchlab/metrics.py` (`vqa_normalize`), `01_metrics.py` |
| VQA accuracy (Figure 2) | `01_metrics.py`, `benchlab/metrics.py` |
| ANLS (Figure 3) | `01_metrics.py`; the function itself is C8's `vlmtrain.metrics.anls` |
| Relaxed accuracy | `01_metrics.py`; C8's `relaxed_accuracy` |
| IoU, Acc@0.5, gIoU and cIoU (Figure 4) | `01_metrics.py`, `benchlab/metrics.py` |
| Yes or no: precision, recall, yes-ratio; CHAIR | `01_metrics.py`, POPE in `11_analyse.py` |
| Elo from pairwise votes | `04_arena.py`, `benchlab/stats.py` |
| The question can be answered without the image (Figure 6, MG and ML) | `11_analyse.py` blind and base-LLM conditions |
| The answer was right but scored wrong | `benchlab/mc.py` extractors, `11_analyse.py` |
| The model prefers a position (Figure 7, CircularEval) | `02_circular_eval.py`, `11_analyse.py` |
| The benchmark is too small to separate models | `03_noise_floor.py`, bootstrap intervals in `11_analyse.py` |
| Input settings (tiling, resolution) | `fmt_notile` condition |
| Reading a results table critically (Figure 9) | the strict / permissive / content and prompt rows above |

## What was not measured

- No result from the post's tables was reproduced. The post's numbers (GPT-4V on BLINK, Qwen2.5-VL on DocVQA, MMStar's
  own filtering counts, Cambrian-1's blind tests) come from papers. This lab runs two 256M and 500M models, which are
  far weaker and may behave differently.
- Only MMStar and POPE were run on real models. ANLS, relaxed accuracy, VQA accuracy, IoU and the mask metrics are
  checked on constructed examples. No VQAv2, DocVQA, ChartQA or RefCOCO data was scored.
- No LLM extractor and no LLM judge were run, so the `content` extractor is a rule-based stand-in. VLMEvalKit's
  choice extractor would rescue more of the `strict` failures. There is no judge-bias experiment.
- MMStar's filtering funnel (22,401 to 11,607 to 1,500) and the contamination detectors were not rebuilt. The lab
  computes MG and ML on the finished benchmark only.
- The SmolLM2 checkpoints stand in for the decoders' pre-VLM weights. They are not the same weights, so ML is only an
  approximation of the paper's quantity.
- The arena is simulated. No human votes, no WildVision data.
- One run per model. Tiling and format-hint differences of 1 to 5 points are close to or inside the 95% intervals
  (about ±2.4 points), and are reported, not tested.
- The CPU-only path of `10_run_models.py` works (tested on 6 items) but a full CPU run was not timed.
