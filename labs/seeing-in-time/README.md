# Lab: Seeing in Time

Code companion for the post [Seeing in Time: Video Understanding in Vision-Language
Models](https://srini-anirudh.github.io/blogs/seeing-in-time/).

The post follows one question through a video model: where should the tokens go? This lab builds the main pieces in
small PyTorch code and runs them on the post's own one-minute clip, where a ball rolls off a table and a person picks up
a cup. You can compute the token budget, pick frames with each sampling rule, cut frames into tubelets, give each token
a time id, spend a fixed budget unevenly and score a predicted moment. Three tiny training runs show what goes wrong
when time is missing: uniform sampling, frame-index positions and a model with no sense of order.

Everything runs on a CPU. Nothing here reproduces the post's paper numbers at model scale. The synthetic videos are
small on purpose, so each effect can be seen in isolation.

## Layout

```
labs/seeing-in-time/
├── seeingtime/
│   ├── budget.py      # tokens per frame, tokens per video, what fits in a context
│   ├── clip.py        # the ball-and-cup events; uniform, fps, Apollo clip, keyframe and query-aware selection
│   ├── synth.py       # the clip rendered in continuous time, plus three tiny labelled video sets
│   ├── tubelet.py     # per-frame patches, depth-2 tubelets (Conv3d), the 2 x 2 merge
│   ├── positions.py   # M-RoPE ids by frame index or absolute time, contiguous vs interleaved frequencies, V2PE
│   ├── allocate.py    # pooling, slow-fast allocation, a merging memory bank, differential patch pruning
│   ├── grounding.py   # temporal IoU, R@1, the best tIoU on a frame grid, the worst-case floor
│   ├── models.py      # bag-of-frames, a tiny video transformer with four time signals, a space-time CNN
│   └── controls.py    # one-frame, shuffled and reversed controls; the T-GRPO temporal bonus; a training loop
├── scripts/           # 01 to 08, one per idea, listed below
├── results/           # what the scripts printed (.txt) and the same numbers as .json
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. No downloads are needed. Two tests compare against the
Hugging Face `transformers` code for Qwen2-VL, Qwen2.5-VL and Qwen3-VL. They build the modules locally and fetch no
weights.

## Run everything

```bash
# Tests: 29 offline tests, plus 2 slow training tests (about 20 s each).
python -m pytest labs/seeing-in-time -m "not slow and not gpu and not network"
python -m pytest labs/seeing-in-time -m slow

# No training, a few seconds each.
python labs/seeing-in-time/scripts/01_token_budget.py
python labs/seeing-in-time/scripts/02_frame_sampling.py
python labs/seeing-in-time/scripts/04_tubelets.py
python labs/seeing-in-time/scripts/06_token_allocation.py
python labs/seeing-in-time/scripts/07_grounding.py
python labs/seeing-in-time/scripts/05_telling_when.py --skip-training   # Part A only

# Tiny training runs on CPU.
python labs/seeing-in-time/scripts/03_playback_speed.py                  # 2 rules x 3 seeds x 600 steps
python labs/seeing-in-time/scripts/05_telling_when.py --steps 600 --seeds 2    # 16 runs
python labs/seeing-in-time/scripts/08_single_frame_shortcut.py           # 3 models x 600 steps
```

Set `OMP_NUM_THREADS` on a shared machine. The results below used 8 threads.

| Post section | Script | What it shows |
|---|---|---|
| The token arithmetic of video (Figure 2) | `01_token_budget.py` | N = D · f · τ for the post's numbers; attention time against MLP time as the sequence doubles |
| Choosing which frames to look at (Figure 3) | `02_frame_sampling.py` | events seen per rule; Apollo clips; keyframes; relevance with and without coverage |
| Choosing frames: the Apollo insight | `03_playback_speed.py` | uniform sampling makes long videos look fast-forwarded |
| Encoding frames, clips and motion (Figure 4) | `04_tubelets.py` | 64 → 32 → 8 tokens; image as two frames; a tubelet sees direction |
| Telling the model when (Figure 5) | `05_telling_when.py` | ids per mode; RoPE scores; frequency layouts; a duration task |
| Fitting a video into the budget (Figure 6, Table 1) | `06_token_allocation.py` | equal vs slow-fast at 1,024 tokens; memory bank; pruning |
| Finding the moment (Figure 9) | `07_grounding.py` | best tIoU per frame rate; the floor; a frame-limited oracle; coarse-to-fine |
| Evaluation and the single-frame shortcut (Figure 10) | `08_single_frame_shortcut.py` | controls; a planted static shortcut; the T-GRPO bonus |

Related code elsewhere in this repository: image M-RoPE ids are in `c07-vlm/tinyvlm/positions.py`, and RoPE itself is
built in `c03-attention` and `c04-llm`. Rewards that require looking are the subject of `c09-vlm-post-training`.

## Results

Every number below is from `results/`. The `.txt` files start with the hardware line and the command.
Hardware: "AMD EPYC 7R13" means AMD EPYC 7R13, 8 threads, torch 2.15.0, CPU only. Every script ran there. The
machine was shared with other jobs, so the attention timings are rough; nothing else depends on load.

Tests on AMD EPYC 7R13: 29 offline tests passed in 20 s, and the 2 slow tests passed in 21 s.
There are no `gpu` or `network` tests.

### Arithmetic and pixels (no training)

| Measurement | Result | Hardware |
|---|---|---|
| Tokens per 448 × 448 frame, 14-px patches, 2 × 2 merge | 256 | computed |
| One hour at 2 fps, 256 tokens per frame | 1,843,200 tokens | computed |
| 1 h at 258 tokens/frame vs 3 h at 66 tokens/frame, 1 fps | 928,800 vs 712,800 tokens | computed |
| SlowFast-LLaVA split, 10 × 288 + 50 × 16 | 3,680 tokens | computed |
| Attention time per sequence doubling, 1K → 8K tokens | ×2.8, ×7.2, ×3.2 (MLP: ×1.8, ×3.7, ×1.8) | AMD EPYC 7R13, shared |
| 8 uniform frames on the clip | 3 of 5 events seen; misses the drop and the cup put down | computed |
| 0.5 fps on the clip | 5 of 5 events, 30 frames, 7,680 tokens; 460,800 tokens for an hour | computed |
| Uniform 16 vs uniform 32 frames | 4 of 5 vs 3 of 5 events: more frames can see fewer short events | computed |
| Keyframes at cosine threshold 0.999, from 120 frames | 29 kept, 4 of 5 events, 11 frames in the roll | computed |
| Top-8 relevance vs relevance + coverage, cup question | spans 40.8-50.8 s vs 13.3-50.8 s | computed |
| Tubelet tokens: per-frame / depth 2 / after 2 × 2 merge | 64 / 32 / 8 | computed |
| Image as two identical frames vs time-summed 2D kernel | max difference 1.6e-6 | AMD EPYC 7R13 |
| Forward vs reversed frame pair, distance between embeddings | 0 for averaged per-frame tokens, 0.91 for a tubelet (norm 11.4) | AMD EPYC 7R13 |
| RoPE score, frame 3 at 1 fps vs frame 3 at 2 fps, frame-index ids | identical (4.7837 and 4.7837) | AMD EPYC 7R13 |
| Longest temporal wavelength, contiguous [16, 24, 24] vs interleaved [24, 20, 20] | 160 ids vs 5,063,256 ids | computed |
| Label contrast kept at 1,024 tokens: equal 16 × 64 / slow-fast (every 4th, 192) / slow-fast (every 8th, 384) | 0.00 / 0.20 / 0.69, all with 4 of 5 events | computed |
| Memory bank, 16 slots, 120 frames | static 21.8-37.8 s merges into one slot of 33 frames; the 1 s put-down keeps 2 slots | computed |
| Differential pruning, 6-px patches | 2.1 % of patches kept; 0 % in static stretches | computed |
| Best tIoU for the Figure 9 moment at 0.25 / 0.5 / 1 / 2 fps | 0.565 / 0.625 / 0.800 / 0.937 | computed |
| Nearest-frame oracle, random 2-10 s events, R@1 at IoU 0.7, 0.25 / 0.5 / 1 / 2 fps | 0.548 / 0.872 / 0.993 / 1.000 | computed |
| Coarse 0.5 fps, then 4 fps in a 6 s window, vs 4 fps everywhere | same tIoU 0.937 with 41 frames instead of 121 | computed |

"Computed" rows are exact arithmetic or deterministic pixel operations; they do not depend on the machine.

### Tiny training runs

All on AMD EPYC 7R13, 8 threads, torch 2.15.0, CPU. Each run takes seconds to a few minutes.

**Playback speed (`03_playback_speed.py`).** A blob moves at 1, 2 or 4 pixels per second; the model sees 8 frames.
Trained on 4-8 s videos, 600 steps, 3 seeds. Chance is 0.33.

| Sampling | Test on 4-8 s videos | Test on 16-32 s videos (unseen) | Share predicted "fast" on long videos |
|---|---|---|---|
| uniform, 8 frames over the video | 0.988 | 0.298 | 0.96 |
| fps, a clip of 8 frames at 2 fps | 1.000 | 1.000 | 0.34 |

Uniform frames from a long video are further apart, so every ball looks fast. This is the mechanism behind Apollo's
finding, at toy scale. Apollo's benchmark gains are not reproduced.

**Duration in seconds (`05_telling_when.py`, Part B).** Videos of 8-24 s, one event of 1-6 s, a 2-layer transformer,
600 steps, 2 seeds. Mean absolute error in seconds. Always predicting the mean gives about 1.25 s.

| Training frame rates | Time signal | Test at 1 fps | Test at 2 fps | Test at 4 fps | Mixed 1/2/4 fps |
|---|---|---|---|---|---|
| mixed 1, 2, 4 fps | none | | | | 0.57 |
| mixed 1, 2, 4 fps | frame index | | | | 0.40 |
| mixed 1, 2, 4 fps | absolute time | | | | 0.19 |
| mixed 1, 2, 4 fps | text timestamp | | | | 0.19 |
| 2 fps only | none | 1.67 | 0.16 | 1.51 | |
| 2 fps only | frame index | 1.60 | 0.16 | 0.57 | |
| 2 fps only | absolute time | 0.32 | 0.16 | 0.11 | |
| 2 fps only | text timestamp | 1.73 | 0.16 | 1.05 | |

With mixed frame rates, a frame index cannot tell a 2-frame event at 1 fps from a 4-frame event at 2 fps. Absolute
time and text timestamps both can. Trained at one frame rate, every model gets 0.16 s, but only absolute time carries
over to new rates. The text-timestamp model ignored its timestamps and counted frames, because at a single rate that
is enough. So in this toy, text timestamps still needed training at several frame rates. That qualifies the post's
point that text timestamps spare the need for data spread over many rates. Qwen3-VL's own training mix is far broader
than this toy.

**The single-frame shortcut (`08_single_frame_shortcut.py`).** 600 steps, seed 0, 3,000 balanced test clips.

| Model | Question | Full | One frame | Shuffled | Reversed |
|---|---|---|---|---|---|
| bag of frames | colour | 1.00 | 1.00 | 1.00 | 1.00 |
| bag of frames | direction | 0.50 | 0.51 | 0.50 | 0.50 |
| bag of frames | order | 0.50 | 0.51 | 0.50 | 0.50 |
| video transformer | colour | 1.00 | 0.92 | 1.00 | 1.00 |
| video transformer | direction | 1.00 | 0.49 | 0.50 | 0.00 |
| video transformer | order | 1.00 | 0.38 | 0.50 | 0.00 |

This is the post's Figure 10 with trained models. Colour survives every control. Direction and order fall to chance
when shuffled and flip to 0.00 when reversed. The video model's colour score drops to 0.92 under the one-frame
control, because eight copies of one frame are an input it never saw. So a control can itself shift the inputs.

| Shortcut-trained video model (red always right, blue always left) | Result |
|---|---|
| Direction accuracy, test with the same shortcut | 1.00 |
| Direction accuracy, balanced test | 0.68 |
| Balanced test, by colour: red / green / blue | 0.49 / 1.00 / 0.51 |
| Share answered "right" for red / blue balls | 1.00 / 0.00 |
| One-frame control on the shortcut test set | 0.77 |

The model reads direction from colour wherever the data let it. On the shortcut test set a single frame already gets
0.77 on a "temporal" question, which is how a benchmark built from static clips overstates temporal skill.

| T-GRPO bonus earned (share of 500 questions, 8 samples each, ordered vs shuffled) | Colour | Direction | Order |
|---|---|---|---|
| bag of frames | 0.00 | 0.41 | 0.39 |
| video transformer | 0.00 | 0.67 | 0.56 |
| video transformer, shortcut-trained | 0.00 | 0.32 | 0.57 |

Colour never earns the bonus, because shuffled frames answer it as well. The bag model's 0.41 and 0.39 come from
sampling noise alone, so the bonus is noisy per question. It separates the models only on average.

## What the numbers mean

- The 8-uniform-frame and 0.5-fps rows are the post's Figure 3 readouts, computed with the same sample times.
  Uniform 32 frames happen to straddle both short events, so they see fewer events than 16 frames. Short events need a
  small gap, not just more frames.
- The tubelet row shows why a depth-2 patch can carry motion: its kernel weights the two frames differently. An encoder
  that runs on each frame alone and averages has no way to tell forward from backward.
- The wavelength row is a derivation from the published section sizes (head dimension 128, RoPE base 1e6). With the
  contiguous layout the time axis owns only the 16 fastest frequencies, so temporal gaps beyond about 160 ids wrap
  around. This is consistent with the Qwen3-VL report's "imbalanced frequency spectrum" explanation. The lab does not
  measure its effect on a benchmark.
- The grounding rows reproduce Figure 9: at 0.5 fps no model can pass IoU 0.7 on that moment. The oracle table is the
  ceiling for any model that answers with frame times.

## What was not measured

- No pretrained video model is run. No benchmark score from the post (Video-MME, MVBench, TempCompass, Charades-STA,
  HourVideo and others) is reproduced, and no number here should be compared with them.
- The paper-scale claims are shown only as mechanisms: Apollo's sampling study, Qwen2.5-VL's absolute-time M-RoPE,
  Qwen3-VL's text timestamps and interleaved MRoPE, SlowFast-LLaVA, MA-LMM, VideoLLaMA 3's pruner and T-GRPO.
- The query-aware selector is a simplified relevance-plus-coverage rule in the spirit of AKS, not AKS itself. Keyframes
  use pixel similarity, not DINOv2 features as in LongVU. Text timestamps are one token built from digit embeddings,
  not a real tokenizer.
- T-GRPO is computed as a reward from sampled answers. No reinforcement learning is run. Its `alpha` and `mu` are this
  lab's settings, not the paper's.
- No GPU was used. Nothing in the post is about GPU speed, and every run fits on a CPU.
