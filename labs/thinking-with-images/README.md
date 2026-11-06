# Thinking with images: a code companion

This lab runs the central claims of the post [Thinking With Images](https://srini-anirudh.github.io/blogs/thinking-with-images/)
on a small open model, Qwen2-VL-2B-Instruct. The post argues that a VLM which looks once cannot think its way back to
pixels the encoder threw away, and that looking again is a form of test-time compute. Here you can watch that happen.

The running example is the post's bakery photo. We render it ourselves, so the answer is known. A 4032 x 3024 street
scene has a large BAKERY sign and a small plaque, `OPEN 9:00-17:30`, whose digits are 32 px tall. A bus-stop sign with
another time on it is the distractor. The question is always "When does the bakery close?"

Six scripts cover six ideas from the post.

| Script | Idea from the post | What it does |
|---|---|---|
| `01_pixel_budget.py` | Where the pixels go | The resolution and token arithmetic, plus a picture of the plaque at each encoder width. No model. |
| `02_lost_detail.py` | Reasoning cannot recover lost pixels | Accuracy against encoder width, with and without longer reasoning, against a crop at the same token cost. |
| `03_probes.py` | Finding the stage that failed | Blind probe, oracle swap, caption-then-reason, crop probe, and a sort of every wrong answer into the post's three failures. |
| `04_views_vs_samples.py` | The sampling trap | Many text samples from one look, against many views of the photo. |
| `05_drift.py` | Long reasoning drifts from the image | Attention on image tokens along a 900-token trace, and what re-injecting a crop does. |
| `06_search.py` | Active perception, search, verification | One look, an attention-guided crop, best-first search over regions, and a verifier that rejects the bus sign. |

## Setup

From the repository root, after `pip install -e ".[dev]"`. The model scripts download Qwen2-VL-2B-Instruct (4.4 GB) from
the Hugging Face Hub on first use. They do not need `torchvision`: `lookagain/vlm.py` does the image preprocessing
itself, which also makes the post's "resolution policy" explicit (resize, cut 14 px patches, merge each 2 x 2 group into
one token).

## Run everything

```bash
# Offline tests: arithmetic, scene rendering, preprocessing, answer parsing, and the probe and search logic on a stub reader.
python -m pytest labs/thinking-with-images -m "not slow and not gpu and not network"

# Tests that load the real model (network, slow).
python -m pytest labs/thinking-with-images -m network

# No model, a few seconds on CPU.
python labs/thinking-with-images/scripts/01_pixel_budget.py

# The model experiments, in order. About 21 minutes on one H100 (the five scripts took 375, 102, 73, 434 and 280 s).
bash labs/thinking-with-images/scripts/run_gpu.sh
# or one at a time:
python labs/thinking-with-images/scripts/02_lost_detail.py --n 100
python labs/thinking-with-images/scripts/03_probes.py --n 100
python labs/thinking-with-images/scripts/04_views_vs_samples.py --n 50 --k 16
python labs/thinking-with-images/scripts/05_drift.py --n 12 --gen-tokens 1024
python labs/thinking-with-images/scripts/06_search.py --n 60
```

The scripts run on CPU too, but a single answer took tens of seconds on a shared CPU node, so we did not try the full
runs there; a code-path check of every model function did run on CPU.

Every model script also takes `--stub`. That swaps Qwen2-VL for `lookagain/stub.py`, a reader that can read text only if
its digits are tall enough in the view it is shown. It exists so the probe, voting and search code can be tested
without the model. Do not read its output as a result. `05_drift.py` has no stub because it reads real attention maps.

## Layout

```
labs/thinking-with-images/
├── lookagain/
│   ├── scene.py     # renders the bakery photos; plaque and bus-sign boxes; oracle facts
│   ├── views.py     # a View is a box of the original photo plus an encoder size; render() crops the original pixels
│   ├── budget.py    # exact token and text-height arithmetic from the post
│   ├── vlm.py       # Qwen2-VL-2B wrapper: own patchify, batching, P(Yes), attention maps
│   ├── answers.py   # prompts and the parser that turns free text into a closing time
│   ├── probes.py    # blind, oracle, caption-then-reason, crop probe, samples against views
│   ├── search.py    # attention-guided crop, best-first search over a region tree, verifier
│   ├── drift.py     # attention share on image tokens along a trace, with re-injection
│   ├── stub.py      # a stand-in reader for offline tests
│   └── util.py      # hardware line and result writer
├── scripts/         # the six experiments and run_gpu.sh
├── tests/           # offline tests, plus real-model tests marked network and slow
└── results/         # real outputs; each starts with a hardware line and the command
```

## Results

Every number names its hardware. Model results are from one run each with fixed seeds, greedy decoding unless stated.
Raw output, including the JSON, is in `results/`. The model is 2B parameters, so its failures are not the failures of the
reasoning VLMs in the post, but the mechanisms are the same.

Hardware for the model scripts: NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2, bf16. Hardware for `01_pixel_budget.py`
and the tests: AMD EPYC 7R13, 8 threads, torch 2.15.0.

### The arithmetic (`01_pixel_budget.py`, AMD EPYC 7R13, 8 threads)

These match the post exactly.

| Quantity | Value |
|---|---|
| Full photo, 14 px patches, no merging | 62,208 visual tokens |
| Resized to 448 x 336 | 768 patch tokens; 32 px text becomes 3.56 px |
| Width for 12 px text | 1512 px, 8,748 patch tokens, 11.4 times the 448 px budget |
| Top-right crop (1008 x 756) resized to 448 x 336 | 768 patch tokens; text is 14.2 px |
| Qwen2-VL merges 2 x 2 patches | the 448 x 336 image is 192 language-model tokens |
| 768 visual and 5,000 reasoning tokens | image is 13.3% of the context |

The script also scores how much of the plaque's letter edges survive each width, as the correlation of intensity
gradients with the original: 0.19 at 448 px, 0.45 at 896 px, 0.73 at 2016 px, and 0.69 for the crop at 448 px.
`results/01_plaque_strip.png` shows the plaque at each width.

### A resolution curve, and what the crop adds (`02_lost_detail.py`, H100)

Direct answers, n = 100 scenes. The second row removes the bus sign from the same photos, so it shows resolution alone.
Chance among the ten possible closing times is 0.10.

| Encoder width (text height, language-model tokens) | 448 (3.6 px, 192) | 672 (5.3 px, 432) | 896 (7.1 px, 768) | 1344 (10.7 px, 1728) | 2016 (16.0 px, 3888) | Top-right crop at 448 (14.2 px, 192) |
|---|---|---|---|---|---|---|
| Accuracy, with bus sign | 0.13 | 0.45 | 0.52 | 0.55 | 0.53 | 0.98 |
| Accuracy, bus sign removed | 0.15 | 0.61 | 0.84 | 0.89 | 0.91 | 0.98 |

- At 448 px the plaque is gone and accuracy is at chance. It rises as text gets taller.
- With the bus sign present, accuracy stops near 0.55. The model answers the bus time in about a quarter of scenes even
  at 2016 px. This is the post's "extracted but not used" failure, and its distractor warning.
- The crop reaches 0.98 at the 448 px token cost. Part of that is resolution and part is that the crop contains no bus
  sign. The second row separates the two: without the bus sign, a full-photo look needs about 1,700 tokens to reach 0.89.
- Chain of thought did not help (n = 40, up to 256 new tokens, mean about 100 tokens generated). Accuracy was 0.07 at
  448 px, 0.45 at 2016 px and 0.57 on the crop. In the saved example at 448 px, the model confidently read a clock that is not in
  the photo. (One saved example is in `results/02_lost_detail.txt`.) This is a small-model result on a short budget. The post's claim is about longer traces from stronger models,
  and this does not test that.
- With no image the model always answers 22:00, not the post's illustrative 18:00. At 448 px it does not answer 22:00
  either (1% of answers). It guesses across many times, so "converges on the language prior" is not what this model does
  at this resolution. The scripts measure the no-image answer and report agreement with it as the `prior` column.

### Finding the stage that failed (`03_probes.py`, H100, n = 100)

| Probe | Accuracy |
|---|---|
| Blind: no image | 0.00 |
| Oracle swap: the facts as text, no image | 0.72 (0.92 when the text omits the bus sign) |
| Crop probe: the plaque cell at 448 px | 1.00 |
| Direct, full photo at 448 px | 0.12 |
| Caption, then reason, at 448 px (closing time in the transcript: 0.07) | 0.13 |
| Direct, full photo at 1344 px | 0.55 |
| Caption, then reason, at 1344 px (closing time in the transcript: 0.87) | 0.85 |

Sorting every wrong direct answer into the post's three failures, in scenes out of 100:

| Encoder width | Correct | 1: never extracted | 2: extracted, not used | 3: reasoning failed |
|---|---|---|---|---|
| 448 | 12 | 57 | 4 | 27 |
| 1344 | 55 | 4 | 25 | 16 |

- Failure 1 dominates at 448 px. At 1344 px it nearly disappears and failure 2 takes over.
- Caption-then-reason at 1344 px scores 0.85, above the 0.55 direct answer, from the same model. Splitting perception
  from reasoning removes most of the bus-sign confusion.
- The oracle swap is below 1.0 because this model mis-copies some half-hour times from text: it turned 19:30 into 19:00
  and 18:30 into 18:00 in the cases we inspected. That is a real reasoning-side failure, and it sets a ceiling for the
  image conditions. The failure-3 column includes it.
- The sort follows the rules at the bottom of `results/03_probes.txt`. A missing time in the transcript is evidence of
  failure 1, not proof.

### The sampling trap (`04_views_vs_samples.py`, H100, n = 50, k = 16)

Text samples are at temperature 1 from one 448 px look. Views are greedy answers on random crops of 25 to 50 percent of
the photo's width, at 448 px each. `wvote` weights each view by the model's own P(Yes) that the view shows opening hours.

| k | Chance pass@k | Text samples: pass / vote | Views: pass / vote / wvote | Visual tokens, views |
|---|---|---|---|---|
| 1 | 0.10 | 0.12 / 0.12 | 0.08 / 0.08 / 0.08 | 192 |
| 4 | 0.34 | 0.26 / 0.14 | 0.24 / 0.04 / 0.20 | 768 |
| 16 | 0.81 | 0.64 / 0.12 | 0.38 / 0.04 / 0.20 | 3072 |

- Voting over text samples does not help: 0.12 at k = 1 and at k = 16. That agrees with the post. The errors are not
  independent of the one encoding.
- It does not reproduce the post's picture of samples converging on one wrong answer. At temperature 1 the model made 9.6
  distinct answers per scene, which is close to guessing. Its pass@16 of 0.64 is below the 0.81 that independent random
  guessing would reach, so pass@k says little here. Read the vote.
- Plain voting over random views is worse (0.04). Most random crops do not contain the plaque, and those crops still
  produce a confident-looking time. Weighting views by the model's own relevance score recovers 0.20, still below one look
  at 2016 px. A view policy needs to know where to look, which is what `06_search.py` adds.

### Attention drift (`05_drift.py`, H100, n = 12, eager attention)

The model is made to write 896 tokens about a 768-token image (end-of-text is suppressed, so late text rambles). One
teacher-forced pass reads, for each generated token, the share of its attention that lands on image tokens, averaged over
heads and layers.

| Generated tokens | 0-127 | 128-255 | 256-383 | 384-511 | 640-767 | 768-895 |
|---|---|---|---|---|---|---|
| Share on image | 0.101 | 0.090 | 0.052 | 0.026 | 0.018 | 0.017 |
| With a crop (192 tokens) inserted after token 256 | 0.101 | 0.090 | 0.085 | 0.037 | 0.023 | 0.022 |
| Image tokens as a share of the context | 0.87 | 0.76 | 0.67 | 0.61 | 0.50 | 0.47 |

- The attention share falls to about a sixth of its starting value. The image's share of the context falls to about half.
  So the drop is faster than dilution alone.
- Inserting the crop lifts the share right after it (0.052 to 0.085), and the lift fades within a few hundred tokens. The
  crop's own tokens count as image tokens in that row.
- This is a teacher-forced measurement on the text the model wrote with one image. A model that had seen the crop would
  write different text. It is not the TVC mid-trace ablation, which measures accuracy.
- The measured share is far below the image's share of the context. We did not separate out attention-sink tokens, so we
  do not say where the rest of the mass goes.

### Active perception, search and verification (`06_search.py`, H100, n = 60, plaque in a random cell)

| Method | Accuracy | Answered the bus time | Model calls | Visual tokens |
|---|---|---|---|---|
| One look at 448 px | 0.15 | 0.23 | 1 | 192 |
| One look at 2016 px | 0.73 | 0.18 | 1 | 3,888 |
| Attention-guided crop: look at 896 px, crop the 4 x 4 cell with the most attention, re-ask at 448 px | 0.85 | 0.08 | 2 | 960 |
| Best-first search over a region tree | 0.88 | 0.05 | 9.9 | 1,894 |
| Best-first search with a verifier | 0.93 | 0.00 | 11.3 | 2,163 |
| Read all 16 cells, keep the answer the verifier trusts most | 0.93 | 0.00 | 32 | 6,144 |

- Looking again beats looking harder. The attention-guided crop beats a 2016 px look at about a quarter of the tokens.
- Attention picks the plaque's cell in 85% of scenes (chance 6%). In the 18 scenes where the 896 px answer was wrong, it
  still picked the plaque's cell in 61%. That fits Zhang and colleagues: the model often knows where to look even when it
  answers wrongly.
- Verification matters. In 3 scenes the search read the bus sign first. The verifier rejected 4 reads over the run and
  lifted accuracy from 0.88 to 0.93.
- The verifier prompt was tuned, and this should be said plainly. A generic prompt ("does it give a shop's opening
  hours, not a bus time?") gave P(Yes) of 0.59 on the bus sign, so it rejected nothing. The prompt used here asks for a sign
  that says OPEN followed by opening hours: 0.91 on the plaque and 0.21 on the bus sign. We chose it on scenes 2000 to
  2039 and ran the search on scenes 5000 and up. The table at the end of `results/06_search.txt` shows both prompts.
- Search cost is in calls as well as tokens: the best-first search with a verifier makes 11 model calls per question.

### Tests

AMD EPYC 7R13, 8 threads, torch 2.15.0: 24 offline tests passed. The 3 real-model tests (`network` and `slow`, run on
CPU in fp32) passed in 36 s. See `results/tests.txt`.

## Which part of the post each script covers

| Post section | Script |
|---|---|
| Perception and reasoning are separate stages; three ways the answer can fail | `03_probes.py` |
| Finding the stage that failed (blind, oracle, caption, crop probes) | `03_probes.py` |
| What reasoning tokens cannot recover; where the pixels go | `01_pixel_budget.py`, `02_lost_detail.py` |
| Long reasoning drifts away from the image | `05_drift.py` |
| The axes of multimodal test-time compute; the sampling trap | `04_views_vs_samples.py` |
| Active perception: looking again | `06_search.py` (attention-guided crop) |
| Searching over visual evidence | `06_search.py` (best-first search) |
| Verifying against the image | `06_search.py` (verifier) |

## What was not measured

- The paper results in the post (V*Bench, MMVP, DeepEyes, Qwen3-VL with tools) were not reproduced. This lab shows the
  mechanisms at small scale on a synthetic photo. Its numbers say nothing about those systems.
- No model was trained. We did not run RL or SFT for zooming, so the post's claims about DeepEyes, GRIT, Pixel Reasoner
  and the cold-start question are not tested.
- Latent visual reasoning, imagined images, drawing tools and visual process reward models are not built.
- The mid-trace image ablation (remove the image halfway through a long trace and measure accuracy) is not run. A 2B model
  writes traces too short for it to say anything, so `05_drift.py` measures attention instead.
- Overthinking and adaptive allocation (stopping rules, length penalties) are not measured.
- One 2B model, one synthetic scene family, one run per setting. Differences of a few points between settings, such as
  0.88 and 0.93, are within what another seed could change at these sample sizes.
