# C9: Post-Train a VLM

Preference learning and RL on a real pretrained vision-language model, `HuggingFaceTB/SmolVLM-256M-Instruct`. The
package `vlmalign` builds synthetic scenes with exact truth, five kinds of hallucination negatives, image-conditioned
DPO written by hand, a check that the preference needs the image, verifiable visual rewards, visual GRPO, a freeze
ablation, and a reward we hack on purpose.

Tutorial: https://srini-anirudh.github.io/blogs/post-train-a-vlm/

The tiny VLM from C7 and C8 is not capable enough for this. Preference learning and RL refine a model that can already
do the task sometimes, so this folder uses a real pretrained model.

## Layout

```
c09-vlm-post-training/
├── vlmalign/
│   ├── scenes.py       # synthetic images with exact facts
│   ├── tasks.py        # questions and sentence templates
│   ├── preference.py   # hallucination negatives, balanced and biased
│   ├── model.py        # SmolVLM loading, prompts, answers
│   ├── logprobs.py     # batches, encode-once path, sequence log-probs
│   ├── dpo.py          # image-conditioned DPO loss and pair batches
│   ├── blind.py        # text-only preference probe
│   ├── rewards.py      # exact match, IoU, Dice, CER, ANLS, field accuracy, weak rewards
│   ├── rl_tasks.py     # RL tasks and the reward registry
│   ├── grpo.py         # rollouts, relative advantages, clipped loss
│   ├── lora.py         # a minimal LoRA
│   ├── freeze.py       # the three freeze modes
│   ├── evaluate.py     # VQA and preference accuracy, real or grey image
│   ├── tiny.py         # tiny random Idefics3 for offline tests
│   └── device.py       # device, threads, seeding
├── scripts/
│   ├── show_pairs.py        # pairs and one batch, tensor by tensor
│   ├── blind_probe.py       # the text-only check
│   ├── reward_audit.py      # hand-written answers against every reward
│   ├── majority_baseline.py # what a constant counting answer scores
│   ├── show_rollouts.py     # one GRPO group printed
│   ├── train_dpo.py
│   ├── train_grpo.py        # freeze ablation and reward hacking
│   ├── perception_probe.py  # did the image features change?
│   └── run_all.sh           # every real run, in order
├── benchmarks/bench_encode_once.py
├── results/                 # real outputs, see below
└── tests/
```

## Run

From the repository root, after `pip install -e ".[dev]"`. Results are written to `c09-vlm-post-training/results/`.
Checkpoints go to `c09-vlm-post-training/runs/`, which git ignores. Scripts use `OMP_NUM_THREADS` threads, or at most 8.

```bash
# Tests
python -m pytest c09-vlm-post-training -m "not slow and not gpu and not network"   # tiny random model, offline
python -m pytest c09-vlm-post-training -m "network"                                # real processor and weights

# Everything below, in order
c09-vlm-post-training/scripts/run_all.sh

# Or one at a time
python c09-vlm-post-training/scripts/show_pairs.py
python c09-vlm-post-training/scripts/blind_probe.py --train 1000 --test 500
python c09-vlm-post-training/scripts/reward_audit.py
python c09-vlm-post-training/scripts/majority_baseline.py
python c09-vlm-post-training/scripts/show_rollouts.py --group 8
python c09-vlm-post-training/benchmarks/bench_encode_once.py --groups 4 8 16
# On a GPU the benchmark also ran G=32 (results/bench_encode_once_gpu.log):
# python c09-vlm-post-training/benchmarks/bench_encode_once.py --groups 4 8 16 32
python c09-vlm-post-training/scripts/train_dpo.py --negatives balanced --steps 150
python c09-vlm-post-training/scripts/train_dpo.py --negatives biased --steps 150
python c09-vlm-post-training/scripts/train_grpo.py --freeze llm --steps 60 --eval both
python c09-vlm-post-training/scripts/train_grpo.py --freeze projector+llm --steps 60
python c09-vlm-post-training/scripts/train_grpo.py --freeze all --steps 60
python c09-vlm-post-training/scripts/perception_probe.py --runs grpo_count_exact_llm grpo_count_exact_projector_llm grpo_count_exact_all
python c09-vlm-post-training/scripts/train_grpo.py --task caption --reward color_recall --steps 60 --max-new-tokens 24 --eval both
python c09-vlm-post-training/scripts/train_grpo.py --task caption --reward color_f1 --steps 60 --max-new-tokens 24
```

## Hardware and what ran where

| Label | What |
|---|---|
| CPU | AMD EPYC 7R13, 8 threads, torch 2.15.0, shared with other jobs, so wall times are noisy |
| H100 | NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2 |

Not every run could use the GPU. The two counting runs that train the language model (`llm`, `projector+llm`) ran on
the H100. The third (`all`) ran on CPU, because the GPU queue was too long. The sampling streams differ between devices,
so read the three counting rows as one experiment with a hardware footnote, not as a controlled timing comparison.
Training seconds below come from each run's JSON, not from queue wait.

| Run | Hardware | Train time |
|---|---|---|
| `dpo_balanced`, 150 steps | CPU | 699 s |
| `dpo_biased`, 150 steps | CPU | 567 s |
| `grpo_count_exact_llm`, 60 steps | H100 | 88 s |
| `grpo_count_exact_projector_llm`, 60 steps | H100 | 159 s |
| `grpo_count_exact_all`, 60 steps | CPU | 934 s |
| `grpo_caption_color_recall_llm`, 60 steps | CPU | 1,014 s |
| `grpo_caption_color_f1_llm`, 60 steps | CPU | 860 s |

## Results

Every number comes from a file in `results/`. All runs are short (60 to 150 update steps), use one seed, and are scored
on small held-out sets (150 preference pairs, 30 questions per kind, 100 counting questions). A difference of a few
points is inside the noise of a set this size. Read these as a working demonstration of each method, not as a
benchmark of SmolVLM.

### Text-only check (CPU, `blind_probe.log`)

| Negatives | Probe accuracy on held-out pairs |
|---|---|
| balanced | 0.490 (chance) |
| biased | 0.914 (colour, count and object 1.00, OCR 0.99, spatial 0.58) |

### Reward audit (`reward_audit.log`)

The weak rewards pay for wrong answers: a full-image box scores 1.00 on `center_in_box` and 0.038 on IoU. Naming five
colours scores 1.00 on recall and 0.57 on F1. The list `1 2 3 4 5` scores 1.00 on `contains` and 0.00 on exact match.

### DPO (CPU, 150 steps, 4 pairs per step, LoRA on the decoder, 4.9M trainable parameters)

Preference accuracy is the fraction of held-out pairs with log pi(chosen) greater than log pi(rejected). The grey
column replaces every image with a grey square. On the balanced evaluation set, a scorer that ignores the image sits at
50%.

| Trained on | Evaluated on | Image before | Image after | Grey before | Grey after |
|---|---|---|---|---|---|
| balanced negatives | balanced pairs | 0.793 | 0.887 | 0.473 | 0.560 |
| biased negatives | balanced pairs | 0.793 | 0.840 | 0.473 | 0.493 |
| balanced negatives | biased pairs | 0.740 | 0.940 | 0.833 | 0.573 |
| biased negatives | biased pairs | 0.740 | 0.960 | 0.833 | 0.900 |

The last row is the shortcut: the biased run raises the grey-image score on biased pairs from 0.833 to 0.900, so most
of what it learned does not need the image. The balanced run lowers that same score to 0.573. Short VQA accuracy
(first word, 30 questions per kind) went from 0.587 to 0.533 after balanced DPO and to 0.520 after biased DPO, so
preference training on these sentences did not improve short answers. Object accuracy fell from 0.767 to 0.233
after balanced DPO.

### Encode-once benchmark (`bench_encode_once.log`, `bench_encode_once_gpu.log`)

One forward and backward pass on one image with G rollouts, all three parts trainable, median of 5.

| G | CPU naive (ms) | CPU once (ms) | CPU speed-up | H100 naive (ms) | H100 once (ms) | H100 speed-up |
|---|---|---|---|---|---|---|
| 4 | 6,922 | 2,081 | 3.33x | 226 | 205 | 1.10x |
| 8 | 12,952 | 2,857 | 4.53x | 286 | 208 | 1.38x |
| 16 | 34,549 | 4,514 | 7.65x | 398 | 209 | 1.90x |
| 32 | not run | not run | not run | 618 | 207 | 2.99x |

The CPU machine was loaded while this ran, so treat its milliseconds as rough. The ratio grows with G because the naive
path repeats the vision encoder G times. On the H100 the encode-once time stays near 207 ms from G = 4 to G = 32, so
at this size the step is limited by something other than arithmetic (launch overhead is the likely cause; we did not
profile it). H100 means NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2.

### Visual GRPO and the freeze ablation (counting, exact-match reward, 60 steps, 4 images x 8 rollouts)

| Trainable | Trainable params | Hardware | Count from pixels | Count from a written list (grey image) | Other kinds, mean |
|---|---|---|---|---|---|
| always answer "1" (majority answer, no model) | 0 | n/a | 0.550 | n/a | n/a |
| before training | 0 | H100 | 0.080 | 0.000 | 0.767 |
| LLM LoRA | 4.9M | H100 | 0.600 | 0.500 | 0.850 |
| connector + LLM LoRA | 12.0M | H100 | 0.550 | 0.000 | 0.825 |
| vision LoRA + connector + LLM LoRA | 14.6M | CPU | 0.520 | 0.000 | 0.767 |

Counting from a written list needs no perception. It rose from 0.000 to 0.500 only in the run where nothing but the
language model could change, so part of that run's gain is counting in language. The two runs that also trained the
connector kept this score at 0.000. The held-out counting questions are not balanced: 55 of 100 have the answer 1, so
always answering "1" scores 0.550 (`results/majority_baseline.txt`). The connector + LLM run's 0.550 equals that rate.
Its training reward in the last 8 steps (0.344) equals the share of the last 32 training images whose answer is 1
(11 of 32), and its groups were all flat from about step 6. A reviewer found that its sampled answers in that range were
all " 1." or " One.". So this run most likely collapsed to a constant answer "1" and did not learn to count. The
LLM-only run's last-8 training reward (0.645) is above the best constant answer on the same 32 images (0.469), but its
held-out 0.600 is only five points above always-"1", which is inside the noise of 100 questions. The pretrained model's
0.080 is below any constant guess because it writes out-of-range numbers, so much of the rise from 0.08 is learning the
answer range and prior. Training reward is whole-text exact match with 12 new tokens, while the held-out score is the
first word with 6 new tokens, so the two are not directly comparable. We saved no per-item predictions, so none of this
could be checked per question.

### Perception probe (CPU, `perception_probe.log`)

A fresh ridge probe (with an intercept) from mean-pooled image features to the count of each shape, fit on 600 scenes
and scored on 200 other scenes (600 shape counts). Same probe, same scenes. Only the features differ.

| Image features from | Per-shape count accuracy |
|---|---|
| pretrained SmolVLM | 0.995 |
| after GRPO, LLM LoRA only | 0.995 |
| after GRPO, connector + LLM | 0.995 |
| after GRPO, vision + connector + LLM | 0.990 |

The probe is at its ceiling on the pretrained features, so it cannot show an improvement. What it does show: the
counts were already linearly readable before any RL, while the model's own counting from pixels started at 0.08. The
LLM-only run cannot change the image features by construction, yet it reaches about the same held-out score as the
runs that can. Most of the rise from 0.08 to about 0.55 is therefore not new information in the features. It is at
least partly the answer range and prior (see the majority-answer row) and partly counting in language (see the
written-list control). This experiment does not separate "counting" from "learning the answer prior". With 600 scored shape counts, 0.990 and 0.995 differ by three counts, which is noise.

### Reward hacking (CPU, caption task, 60 steps, 4 images x 8 rollouts, evaluated on 50 scenes)

| Reward trained on | Colours named per answer | Colours not in the image per answer | Recall | Honest F1 |
|---|---|---|---|---|
| before training | 1.90 | 0.06 | 0.742 | 0.765 |
| recall (weak) | 5.46 | 2.98 | 1.000 | 0.703 |
| F1 | 2.48 | 0.00 | 1.000 | 1.000 |

The weak reward reaches its maximum by naming every colour. Its training reward went from 0.881 to 1.000 while the
honest F1 monitor went from 0.846 to 0.628 (means over the first and last 8 steps). The F1 reward cannot be gamed this
way, and the model learns to name only the colours that are there.

## Not yet measured

- The three counting runs on one device. A single-device rerun of `grpo_count_exact_all` on a GPU would remove the
  hardware footnote in the ablation table.
- Anything on more than one GPU. Nothing in this folder needs more than one.

## Fixes made while running

- `scripts/perception_probe.py`: the ridge probe had no intercept, so on centred features it predicted values near zero
  for counts whose mean is about 1.3, and scored 0.000 on features that held the answer. It now fits the mean count as
  the intercept. `tests/test_probe.py` checks both directions.
- `vlmalign/device.py`: `set_threads()` replaces a fixed two-thread setting, so one environment variable controls all
  scripts.
- `vlmalign/lora.py`: `lora_delta_norms` detaches before `float()`, which removes a warning.
