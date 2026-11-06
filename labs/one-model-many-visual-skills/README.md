# Lab: One Model, Many Visual Skills

Code companion to the post [One Model, Many Visual Skills](https://srini-anirudh.github.io/blogs/one-model-many-visual-skills/).
The post explains multimodal post-training: how one set of weights learns to answer, describe, count and reason about
images, and what it can lose on the way. This lab makes those ideas happen in a model small enough to train on a CPU.

The model is a tiny VLM written from scratch, with 238K parameters. It has a one-block patch encoder, an MLP connector
and a three-layer causal decoder. The images are synthetic scenes: a 3 x 3 grid of cells, each empty or holding a
plate, fork, cup, ball or book. Each cell is one patch, so an image is 9 visual tokens. Before any image is seen, the
decoder is trained on a text-only skill: reverse a list of words. That is its "language ability". Multimodal training
can damage it, as it can damage the language model inside a real VLM.

Every number here comes from this toy setting. None of the post's paper results were reproduced at their scale. That
includes LLaVA-1.5, Cambrian-1, NVLM, MM1.5 and the others. The lab shows the mechanisms the post describes, small enough
to inspect.

## Layout

```
labs/one-model-many-visual-skills/
├── skillmix/
│   ├── scenes.py     # the scene world, rendering, the plate-fork co-occurrence, counterfactual repainting
│   ├── chat.py       # word vocabulary, ChatML-style template, <image> expansion, the loss mask
│   ├── tasks.py      # instruction sources: vqa, chat, caption (clean or biased teacher), exist, reason, text
│   ├── model.py      # TinyVLM (encoder, connector, decoder) and LoRA
│   ├── train.py      # mixture batches, per-token and per-example loss, freezing strategies, greedy decoding
│   ├── evaluate.py   # the capability vector: one held-out suite per skill, blind and counterfactual variants
│   ├── mixture.py    # per-source caps, shares in examples, compute and loss tokens, the Pareto front
│   └── runs.py       # the shared text-pretrained base model, parallel sweeps, result files
├── scripts/          # 01 to 07, one per idea, listed below
├── results/          # outputs written by the scripts; each file starts with its hardware and command
└── tests/
```

Related tutorials in this repository build the general machinery at a larger scale. C6 (`c06-post-training`) builds SFT
with loss masking on a language model. C7 (`c07-vlm`) builds a VLM from pretrained parts. C8 (`c08-vlm-training`)
trains it with task mixtures and text replay. This lab does not reuse their code. It needs a model small enough to run
about forty-five training runs on a CPU, so that each claim can be tested with a sweep.

## Setup

From the repository root, after `pip install -e ".[dev]"`. Nothing is downloaded. The scripts train on CPU.

## Run everything

```bash
python -m pytest labs/one-model-many-visual-skills -m "not slow and not gpu and not network"

python labs/one-model-many-visual-skills/scripts/01_anatomy.py              # seconds, no training
python labs/one-model-many-visual-skills/scripts/02_loss_normalization.py   # 6 runs
python labs/one-model-many-visual-skills/scripts/03_format_conflict.py      # 10 runs
python labs/one-model-many-visual-skills/scripts/04_forgetting.py           # 6 runs
python labs/one-model-many-visual-skills/scripts/05_teacher_hallucination.py  # 3 runs
python labs/one-model-many-visual-skills/scripts/06_vision_dependence.py    # 1 run
python labs/one-model-many-visual-skills/scripts/07_pareto.py               # 20 runs
```

The first script that needs it trains the base text model (1,500 steps) and caches it in `runs/`, which git ignores.
Sweeps run in parallel worker processes. `LAB_WORKERS` (default 4) and `LAB_THREADS_PER_JOB` (default 2) set how many.
Every run starts from the same base checkpoint, uses batch 32 and AdamW at 2e-3 with cosine decay, and trains 1,000 steps
unless the table says otherwise. Evaluation suites have 200 to 300 held-out items, decoded greedily.

## Which script covers which part of the post

| Post section | Script | What you see |
|---|---|---|
| Anatomy of an instruction example; Which tokens receive loss | `01_anatomy.py` | the serialized two-image chat and its loss mask; position and loss-token counts |
| Multiple images and multiple turns (the normalization nuance) | `01_anatomy.py`, `02_loss_normalization.py` | the post's 1.3% vs 50% figure, then the same mix measured and trained |
| The multimodal task mixture (caps; examples vs compute vs loss tokens) | `01_anatomy.py`, part 4 | three shares of one made-up collection, before and after a per-source cap |
| Negative transfer: short answers against conversation | `03_format_conflict.py` | answers shrink to one word as the VQA share grows; the format prompt removes the conflict |
| Language forgetting; Strategies against forgetting | `04_forgetting.py` | Delta S_text and the multimodal score for replay, lower LR, LoRA and a frozen decoder |
| A VLM teacher; Detailed captions and short answers (LURE, POPE) | `05_teacher_hallucination.py` | a teacher's invented fork reaches the student; verification removes it |
| Does the rationale actually use the image?; Use benchmarks that need the image | `06_vision_dependence.py` | blank-image and swapped-image scores; a counterfactual repaint changes the rationale |
| Evaluating while training (the Pareto view) | `07_pareto.py` | a real Pareto front over 10 mixtures, with and without the format prompt |

## Results

Hardware for every row: AMD EPYC 7R13, torch 2.15.0, CPU only. Sweeps ran as 6 worker processes with 1 thread each
(script 06: 1 process, 4 threads). One training run took 30 s to 145 s, depending on steps and machine load. The
base text model took 110 s at 8 threads. Each `results/*.txt` file starts with its hardware and command. Everything is
one seed except where the row says two.

### 1. Anatomy and the loss mask (`01_anatomy.py`, no training)

| Measurement | Result |
|---|---|
| Two-turn, two-image chat, 9 visual tokens per image | 54 positions, 18 visual, 15 carry loss (27.8%) |
| The same chat at LLaVA-1.5's 576 tokens per image (arithmetic) | 1,188 positions, 1,152 visual, 15 carry loss (1.3%) |
| VQA half's share of the signal, post's example (2 vs 150 loss tokens), per-token / per-example | 1.3% / 50.0% |
| Same, measured on 20 batches of this lab's data (2.2 vs 27.0 loss tokens) | 6.8% / 50.3% |

The mask covers each assistant answer and its `<|im_end|>`, and nothing else. The 1,188 figure matches the post's
"about 1,190 positions" for Figure 1. Part 4 of the output shows that, in a made-up collection, a source with 70% of the
examples can hold 47% of the compute and 4% of the loss tokens.

### 2. Per-token vs per-example averaging (`02_loss_normalization.py`)

Mix: 50% one-word VQA and 50% detailed captions, by examples.

| Run | VQA exact | Caption recall | CHAIR_i |
|---|---|---|---|
| per-token, 250 steps | 0.477 | 0.425 | 0.284 |
| per-example, 250 steps | 0.450 | 0.429 | 0.319 |
| per-token, 1,000 steps, seed 0 | 0.733 | 0.552 | 0.190 |
| per-example, 1,000 steps, seed 0 | 0.667 | 0.450 | 0.287 |
| per-token, 1,000 steps, seed 1 | 0.713 | 0.416 | 0.239 |
| per-example, 1,000 steps, seed 1 | 0.700 | 0.459 | 0.191 |

This is a negative result. The normalization moves the VQA share of the signal from 6.8% to 50%, as part 1 measured.
Yet across two seeds the scores do not move by more than the seed-to-seed spread. With seed 0, per-token looks better
on both skills. With seed 1, per-example has the better recall. In this toy, short answers are easy enough that 6.8% of
the signal is plenty. The post's advice stands as a bookkeeping rule: state the mixture in loss tokens, because the
normalization changes it. This lab does not show which choice trains better.

### 3. The format conflict (`03_format_conflict.py`)

VQA and chat ask the same questions. Evaluation asks unmarked questions. "Long" means an answer of 6 words or more.

| VQA share | Long rate, no prompt | Mean words, no prompt | Long rate, format prompt | VQA exact, no prompt | VQA exact, format prompt |
|---|---|---|---|---|---|
| 0.00 | 1.000 | 15.02 | 1.000 | 0.000 | 0.000 |
| 0.25 | 1.000 | 15.03 | 1.000 | 0.003 | 0.597 |
| 0.50 | 0.900 | 13.53 | 1.000 | 0.083 | 0.687 |
| 0.75 | 0.070 | 2.09 | 1.000 | 0.677 | 0.723 |
| 0.90 | 0.000 | 1.17 | 1.000 | 0.630 | 0.723 |

Without a marker, one style wins and the other is lost. Below a VQA share of one half the model answers in sentences,
so it fails exact-match VQA. Above it, a user's ordinary question gets a one- or two-word reply, the post's "A bus."
case. The switch is
sharper than the smooth curve in the post's Figure 4: this tiny model mostly picks the majority style instead of
averaging lengths. With the format prompt, every run keeps long answers for unmarked questions (long rate 1.000)
and still answers marked questions in one word. The fix is conditioning. No data was removed.

### 4. Language forgetting and the levers (`04_forgetting.py`)

The base model scores 1.000 on the text suite (reverse a word list, exact match). Multimodal score = mean of VQA exact,
chat correct and caption recall.

| Run | Text | Delta S_text | VQA | Chat correct | Recall | Multimodal score |
|---|---|---|---|---|---|---|
| full fine-tune, replay 0% | 0.003 | -0.997 | 0.730 | 0.733 | 0.451 | 0.638 |
| full fine-tune, replay 10% | 1.000 | +0.000 | 0.740 | 0.733 | 0.596 | 0.690 |
| full fine-tune, replay 50% | 1.000 | +0.000 | 0.583 | 0.607 | 0.522 | 0.571 |
| decoder at 0.1x learning rate, replay 0% | 0.980 | -0.020 | 0.560 | 0.513 | 0.578 | 0.550 |
| LoRA rank 8 on the decoder, replay 0% | 0.000 | -1.000 | 0.557 | 0.633 | 0.540 | 0.577 |
| frozen decoder, replay 0% | 1.000 | +0.000 | 0.000 | 0.113 | 0.000 | 0.038 |

Without replay, 1,000 multimodal steps erase the text skill completely. Ten percent replay keeps it at 1.000, and the
multimodal score is also the highest of all runs (0.690 against 0.638). That matches the direction VILA reports. It is
one seed, so read it as "replay did not cost visual skill", not as proof that replay helps. Half replay keeps text but
leaves less budget for the visual skills. A lower decoder learning rate protects text almost fully and learns less.
A frozen decoder protects text exactly but cannot learn the new answer formats at all, which is the post's NVLM point
in miniature. LoRA did not protect text here: rank 8 on every decoder linear layer, at the full learning rate, is
enough capacity to overwrite a skill this small. "LoRA learns less and forgets less" is a finding about large models,
and this lab does not reproduce it.

### 5. A teacher's hallucination reaches the student (`05_teacher_hallucination.py`)

The biased teacher adds "there is a grey fork at ..." as the last sentence, 90% of the time, when a plate has no fork.
That bias is built in by hand to mirror POPE and LURE. It is not a reproduction of them. Evaluation scenes hold a plate
and no fork.

| Student trained on | Fork rate in captions | CHAIR_i | Recall | "is there a fork ?" -> yes |
|---|---|---|---|---|
| clean annotator captions | 0.000 | 0.147 | 0.436 | 0.047 |
| biased teacher captions | 0.830 | 0.432 | 0.464 | 0.000 |
| teacher captions after the object check | 0.000 | 0.216 | 0.546 | 0.000 |

The student copies the teacher's co-occurrence bias almost exactly: 83% of its captions name the absent fork. The
object-level check, which drops every sentence naming an object the scene lacks, removes it completely. One nuance is
easy to miss. The yes/no probe stays clean for the biased student (0.000 yes), because its yes/no questions were
built from the true scene. The hallucination lives in the caption format only. So a POPE-style probe alone would have
missed it. CHAIR_i is above zero even for the clean student because this tiny model's perception is weak: its
captions repeat or confuse objects (recall below 0.55).

### 6. Does the answer need the image? (`06_vision_dependence.py`, 2,000 steps)

"Blank" removes every object. "Swap" shows an unrelated scene that still holds the object the question names.

| Suite | Real image | Blank | Swap | Real minus swap |
|---|---|---|---|---|
| VQA (colour, count, position) | 0.730 | 0.140 | 0.213 | +0.517 |
| Fork colour (always grey, so the text answers it) | 1.000 | 0.000 | 1.000 | +0.000 |
| Reasoning with a rationale ("red 2 . blue 1 . 2 minus 1 is 1 .") | 1.000 | 0.000 | 0.330 | +0.670 |

| Counterfactual check: repaint one blue object red and ask again | Result |
|---|---|
| Rationale exactly right for the edited image | 1.000 |
| Rationale identical before and after the edit | 0.000 |

The fork-colour suite scores the same with the wrong image as with the right one. It cannot see any change in visual
skill, which is the post's warning about benchmarks with a small vision gap. The blank-image test does not reveal
this: an empty scene is out of distribution, so the model fails every suite on it, even the one that needs no image.
The swap test is the cleaner probe. The reasoning model reads the image: every rationale changes its counts after
the repaint and matches the new scene.

### 7. The Pareto view, with real runs (`07_pareto.py`)

Ten random mixtures over VQA, chat, captions and text replay, each trained twice: without and with the format prompt.
The two axes are VQA exact and the long-answer rate on unmarked questions.

| | Without format prompt | With format prompt |
|---|---|---|
| Mixtures on the Pareto front | 6 of 10 | 1 of 10 |
| Long-answer rate across the 10 mixtures | 0.000 to 1.000 | 1.000 in every run |
| Best VQA exact | 0.703 (long rate 0.000) | 0.733 (long rate 1.000) |
| Weighted sum, weights (0.8, 0.2) / (0.5, 0.5) / (0.2, 0.8) | three different mixtures | the same mixture |
| Best VQA with long rate >= 0.9 and text >= base - 0.01 | VQA 0.003 | VQA 0.733 |

Without the prompt the two skills compete, and the front is a real trade-off curve. The choice of weights decides the
model. The constraint "keep conversational answers" forces VQA down to 0.003. With the prompt, the conflict is gone
and one mixture beats all others on both axes. Fixing the conflict moved the whole front outward, which no reweighting
could do.

## Tests

```bash
python -m pytest labs/one-model-many-visual-skills -m "not slow and not gpu and not network"
```

18 tests, about 10 s on AMD EPYC 7R13. There are no `slow`, `gpu` or `network` tests: nothing is downloaded and
everything runs on CPU. The tests check the loss mask, placeholder expansion and vocabulary coverage, the scene rules,
the teacher bias and the verifier, both loss normalizations against hand-computed values, the causal reach of the
image, LoRA's identity start, which parameters each strategy trains, a short training run, and the Pareto helpers.

## What was not measured

- No result from the post's papers was reproduced at scale. Every model here has 238K parameters and sees synthetic
  24 x 24 scenes.
- Most rows are one seed. Script 02 shows that seed noise can be as large as the effect under study, so treat
  single-seed differences of a few points as noise.
- The text skill is one narrow task (reversing word lists), not MMLU, GSM8K or HumanEval. Safety forgetting, multi-image
  binding, video, OCR, grounding and multilingual data were not built.
- Vision-encoder forgetting and probes of the encoder before and after tuning were not measured.
- Staged curricula against a joint mixture were not compared.
- The teacher's bias was written by hand. The lab shows that a student copies such a bias, not how often real teachers
  have it.
- No GPU was used. The post is about data, not speed, and every experiment fits on a CPU.
