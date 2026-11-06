# Lab: Rewarding What the Model Sees

Code companion for the post [Rewarding What the Model Sees](https://srini-anirudh.github.io/blogs/rewarding-what-the-model-sees/),
on reinforcement learning with verifiable rewards for vision-language models.

The post makes claims about visual rewards that are small enough to test on a CPU. This lab tests them. It builds the
post's verifiers and scores the cheapest degenerate output of each. It computes the GRPO arithmetic the post quotes.
It trains a box policy to see whether a soft IoU reward makes boxes grow. And it trains a tiny chart-reading VLM from
scratch, to compare outcome rewards with step-level checks, three trainable sets, and plain with dynamic sampling.
Every chart run reports the post's four controls before and after RL.

Nothing here reproduces a paper's result at scale. The papers trained 2B to 72B models on real data. This lab shows the
mechanisms the post describes, on models small enough to train in seconds.

## What is reused

[C9: Post-Train a VLM](../../c09-vlm-post-training/) already writes exact match, box IoU, mask IoU and Dice, CER and
ANLS, and runs visual GRPO with a freeze ablation on SmolVLM-256M. This lab loads C9's `rewards.py` for those
primitives (`visrl/_c09.py`) and does not rebuild them. Everything else is new here.

## Layout

```
labs/rewarding-what-the-model-sees/
├── visrl/
│   ├── _c09.py        # loads C9's reward primitives by path
│   ├── verifiers.py   # numeric tolerance, GIoU, coverage, thresholds, points, matching,
│   │                  # the category-list hack and its fix, strict and lenient parsers, JSON gate x F1
│   ├── grpo.py        # group advantages, zero-advantage probability, difficulty filter
│   ├── boxes.py       # the box-size policy and vectorised box rewards
│   ├── charts.py      # two-bar charts, the tiny VLM (encoder, connector, LM), GRPO, the four controls
│   └── report.py      # results files with a hardware line and the command
├── scripts/
│   ├── audit_rewards.py       # cheapest output vs honest miss, for every reward
│   ├── advantage_math.py      # near ties, the whole-image box in a group, p^K + (1-p)^K
│   ├── credit_assignment.py   # prod of q_i, closed form and Monte Carlo
│   ├── train_box_policy.py    # GRPO on box size under six rewards
│   ├── train_chart_policy.py  # --train-only / --report-only / --only, see Run everything
│   └── run_all.sh
├── results/           # real outputs, each file starts with the hardware and the command
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Only PyTorch is needed. No downloads, no GPU.
Set `OMP_NUM_THREADS` on a shared machine.

## Run everything

```bash
# Tests (about 10 s)
python -m pytest labs/rewarding-what-the-model-sees -m "not slow and not gpu and not network"

# Every result in results/, in order. The box, audit and arithmetic scripts take under a minute in total.
# The chart training is the long part: one seed trains six configs. Run the seeds in parallel, see below.
labs/rewarding-what-the-model-sees/scripts/run_all.sh

# Or one at a time
python labs/rewarding-what-the-model-sees/scripts/audit_rewards.py
python labs/rewarding-what-the-model-sees/scripts/advantage_math.py
python labs/rewarding-what-the-model-sees/scripts/credit_assignment.py
python labs/rewarding-what-the-model-sees/scripts/train_box_policy.py --steps 300 --seeds 0 1 2

# The chart policy: train each seed (the three experiments share their configs), then write the three reports
for s in 0 1 2; do
  python labs/rewarding-what-the-model-sees/scripts/train_chart_policy.py --seeds $s --train-only --threads 3 &
done; wait
python labs/rewarding-what-the-model-sees/scripts/train_chart_policy.py --seeds 0 1 2 --report-only
```

The chart runs in `results/` took 20 to 25 minutes per seed with 3 threads per seed on a machine that other jobs were
using, three seeds in parallel. A short trial on 4 threads suggests about 11 minutes per seed on an idle 8-thread CPU.
That trial was not repeated for the full run. Per-seed logs go to `runs/`, which git ignores. The matched-rollout
baseline was added after the first runs, with `--only "96 prompts"`, which trains only configs whose label contains that
text and merges them into the same `runs/` file.

## Results

All numbers below are from `results/`. Hardware for every row: AMD EPYC 7R13 CPU, CPU only, torch 2.15.0. The thread
count of each run is on the first line of its results file. Several runs shared the machine with other jobs, so the
wall times are not benchmarks.

### Cheap outputs against honest misses (`audit_rewards.txt`)

The target covers 20% of the image. The honest miss is a box of similar size placed just above it.

| reward | cheap output | cheap output scores | honest miss scores | cheap output wins? |
|---|---|---|---|---|
| soft IoU | whole-image box | 0.200 | 0.000 | yes |
| IoU >= 0.5 | whole-image box | 0.000 | 0.000 | no |
| GIoU | whole-image box | 0.200 | -0.333 | yes |
| coverage | whole-image box | 1.000 | 0.000 | yes |
| centre hit | whole-image box | 1.000 | 0.000 | yes |
| IoU + size term | whole-image box | 0.200 | 0.375 | no |
| multiple choice, lenient parser | "B or C" (gold C) | 1.000 | 0.000 | yes |
| multiple choice, strict parser | "B or C" (gold C) | 0.000 | 0.000 | no |
| numeric 5%, first number | "between 56 and 60" (gold 58) | 1.000 | 0.000 | yes |
| numeric 5%, exactly one number | "between 56 and 60" (gold 58) | 0.000 | 0.000 | no |
| category score, COCO-style | a box for all 8 listed categories | 1.000 | 1.000 | ties |
| category score, length-aware | a box for all 8 listed categories | 0.125 | 0.500 | no |
| JSON field recall | every schema field guessed | 1.000 | 0.667 | yes |
| JSON schema gate x F1 | every schema field guessed | 0.545 | 0.800 | no |

The same file checks the box-section arithmetic. GIoU equals IoU for every containing box. A containing box passes
IoU >= 0.5 at 1.9 times the target's area (IoU 0.526) and fails at 2.1 times (0.476). GIoU ranks two misses that IoU
scores the same: a near miss gets -0.013 and a far miss -0.573. A misreading of 56 for 58 passes the 5% band.

### GRPO arithmetic (`advantage_math.txt`, `credit_assignment.txt`)

| quantity | post says | computed |
|---|---|---|
| 7 rollouts at IoU 0.30, one at 0.31: advantages | about +2.6 and -0.4 | +2.645 and -0.378 (population std); +2.475 and -0.354 with the sample std |
| same group without the std division (Dr. GRPO) | near-ties not amplified | +0.009 and -0.001 |
| 7 misses and one whole-image box: group mean | 0.025 | 0.025; the box gets +2.646, each miss -0.378 |
| P(zero advantage), K = 8, p = 0.5 / 0.9 / 0.95 / 0.05 | 0.8% / 43% / 66% / 66% | 0.78% / 43.0% / 66.3% / 66.3% (Monte Carlo agrees to within 0.4 points) |
| credit of one of 5 steps, each right with q = 0.8 | 0.8^4 = 0.41 | 0.410 (Monte Carlo 0.409) |

A batch of 200 prompts with pass rates spread around 0.9 has only 37% useful groups at K = 8. Filtering to 0.2-0.8 on
pass rates estimated from 16 rollouts keeps 31 prompts, and 97% of their groups are useful. The table for means 0.3,
0.5 and 0.7 is in the file.

### Box policy under six rewards (`box_policy.txt`)

GRPO trains only the box size. Each rollout's centre has location noise: 0.01 image widths on easy targets, 0.15 on hard
ones. The policy starts with boxes the target's size. After 300 steps, mean [min, max] over 3 seeds:

| reward | box area / target area, easy | box area / target area, hard | useful groups, hard |
|---|---|---|---|
| soft IoU | 1.01 [1.01, 1.02] | 2.20 [2.12, 2.32] | 1.00 |
| soft IoU, no std division | 1.01 [1.01, 1.01] | 1.97 [1.91, 2.00] | 1.00 |
| IoU >= 0.5 | 0.98 [0.96, 1.02] | 1.17 [1.15, 1.20] | 0.47 |
| GIoU | 1.01 [1.01, 1.02] | 4.41 [4.25, 4.56] | 1.00 |
| coverage | 9.39 | 108 (the box is the whole image) | 0.00 |
| centre hit | 1.00 | 12.71 [12.62, 12.84] | 0.95 |
| 0.5 IoU + 0.5 size match | 1.02 [1.01, 1.02] | 1.09 [1.07, 1.10] | 1.00 |

Soft IoU doubles the box on hard targets and leaves easy ones alone. That is the GUI-G1 effect in miniature: when the
policy cannot locate the target, a bigger box earns more expected IoU. Dropping the std division does not remove it,
so the growth comes from the reward, not from the normalisation. GIoU grows boxes even more, because on a miss its
penalty shrinks as the box grows. The threshold keeps boxes near the target's size, at the cost of half the hard groups
carrying no signal. The size term holds the box at 1.09 times the target.

Two rows need a note. Under coverage every box soon covers the whole image, all rewards are 1, and learning stops. The
ratio 108 is the policy's unclipped size parameter at that point. The centre-hit reward does not depend on size at all,
yet boxes grew 12 times. The cause is clipping: a huge box clipped to the image has its centre pulled toward the image
centre, where these targets lie. GUI-G1 reported that a hit reward shrinks boxes. This toy cannot show that, because
here size never changes where the centre lands except through clipping.

### Tiny chart-reading VLM (`chart_credit`, `chart_trainable`, `chart_dynamic`)

The model reads a two-bar chart (bar values 0 to 9, random shift, contrast and noise) and a question, SUM or absolute
DIFFERENCE, then writes three tokens: reading 1, reading 2, answer. It has a CNN encoder, a linear connector and a
2-layer causal transformer, trained from scratch. It has 79,318 parameters: 4,800 in the encoder, 2,112 in the connector and 72,406 in the LM and head. An SFT warm start (150 steps on 300 charts)
gives a model that is right about half the time. GRPO then runs for 200 steps with 32 charts and K = 8 rollouts per
step, learning rate 1e-3, no KL term. All numbers are the mean over 3 seeds, with [min, max] after RL, on 1000
held-out charts, greedy decoding.

The four columns are the post's controls. `A_img` is answer accuracy on the real image. `A_blank` replaces the image
with a blank one. `A_oracle` writes the true readings into the answer and lets the model generate only the answer.
`P` is the share of charts where both readings are right.

**Outcome reward against step-level checks** (the post's credit-assignment section)

| reward | A_img | A_blank | A_oracle | P | rewarded rollouts with a wrong reading, first 20 -> last 20 steps |
|---|---|---|---|---|---|
| before RL | 0.505 | 0.067 | 0.562 | 0.793 | |
| outcome (answer right) | 0.627 [0.522, 0.764] | 0.067 | 0.634 [0.560, 0.719] | 0.725 [0.558, 0.838] | 17% -> 17% |
| step checks (each reading, and the subtraction against the model's own readings) | 0.710 [0.699, 0.718] | 0.052 | 0.729 [0.712, 0.738] | 0.949 [0.933, 0.961] | 10% -> 1.5% |

Under the outcome reward the answer improves while the readings get slightly worse on average (P falls from 0.793 to
0.725, and one seed ends at 0.558). About one rewarded rollout in six has a wrong reading, and that share does not
shrink. Step checks push P up to 0.95 and drive the lucky share to 1.5%. The step reward also sees the true bar values,
so it uses an annotation the outcome reward does not use. That is the cost the post names. The step-reward seeds also
agree with each other more closely (A_img range 0.02, against 0.24 for the outcome reward), but three seeds are few.

**Which parameters receive the gradient** (the post's parameters section)

| trainable set | A_img | A_oracle | P | encoder linear probe, before -> after | mean grad norm: encoder / connector / LM |
|---|---|---|---|---|---|
| LM only | 0.680 [0.648, 0.718] | 0.689 | 0.752 | 0.984 -> 0.985 | 0 / 0 / 5.11 |
| connector + LM | 0.627 [0.522, 0.764] | 0.634 | 0.725 | 0.984 -> 0.984 | 0 / 2.88 / 5.03 |
| full model | 0.673 [0.623, 0.730] | 0.639 | 0.743 | 0.984 -> 0.987 | 2.79 / 2.86 / 5.00 |

The three sets cannot be told apart here. The seed ranges overlap, and the encoder probe moves by at most 0.003. This
does not reproduce the post's reported findings. V-Triune's encoder instability does not appear, but only the mean
gradient norm was recorded, not its spikes. The preference-training encoder gain does not appear either. The encoder
here is small and already reads the bars well after SFT (probe accuracy 0.984), so there is almost nothing left for RL
to improve. The experiment shows that the freeze switch works and that gradients stop where the post says. It does not
show which choice is better.

**Dynamic sampling** (the post's zero-advantage section)

| sampling | A_img | A_oracle | P | rollouts generated |
|---|---|---|---|---|
| plain, 32 charts per step | 0.627 [0.522, 0.764] | 0.634 | 0.725 | 51,200 |
| dynamic: draw 96, keep up to 32 groups with non-zero advantage | 0.772 [0.719, 0.805] | 0.734 | 0.772 | 153,600 |
| plain, 96 charts per step (same rollouts as dynamic) | 0.834 [0.784, 0.874] | 0.805 | 0.837 | 153,600 |

Dynamic sampling beats the 32-chart baseline, but it also generates three times the rollouts. A plain run that uses the
same number of rollouts does better still. In this toy the gain comes from the larger sample, not from discarding
zero-advantage groups. The learning-rate and batch size were not tuned for any of the three. The useful-group share of
the raw 32-chart batch falls from about 0.6 to 0.25-0.4 as the policy improves in all three runs, which is the
vanishing-advantages effect the post describes.

## Post sections and the scripts that cover them

| post section | what is checked | script |
|---|---|---|
| A VLM is already a policy | group baseline holds the image fixed; equal groups give zero advantage | `advantage_math.py`, `tests/test_rl.py` |
| A taxonomy of visual rewards | each verifier written and tested; the near-tie amplification note | `visrl/verifiers.py`, `advantage_math.py` |
| Designing a box reward | the 0.2 floor, the 0.025 group mean, threshold, GIoU and coverage behaviour, box growth under training | `audit_rewards.py`, `train_box_policy.py` |
| How visual rewards get hacked | oversized box, list-every-category, lenient parsers, JSON guessing | `audit_rewards.py` |
| Which parameters receive the gradient | the three trainable sets, gradient norms, encoder probe | `train_chart_policy.py` (`trainable`) |
| Perception RL and reasoning RL | blank-image, oracle and probe controls | `train_chart_policy.py` (all) |
| Credit assignment under a sparse reward | credit = product of the other steps; cancelling misreadings | `credit_assignment.py`, `train_chart_policy.py` (`credit`) |
| Process rewards for visual reasoning | step-level checks against the outcome reward | `train_chart_policy.py` (`credit`) |
| Curriculum and the zero-advantage problem | p^K + (1-p)^K; difficulty filter; dynamic sampling | `advantage_math.py`, `train_chart_policy.py` (`dynamic`) |

## What was not measured

- No paper's result is reproduced. The box policy, the chart VLM and the pass-rate batches are toys that show the
  mechanisms. The post's numbers for GUI-G1, VLM-R1, V-Triune, Skywork, VL-Rethinker and the rest are not tested.
- The box policy learns one scalar, the box size. It says that the soft IoU reward pays a bigger box on a hard target.
  It does not say how a real VLM's boxes would change.
- The chart VLM never saw natural images, so the claims about natural-image grounding, OCR, tables, masks and
  documents are not tested. Those verifiers are tested as functions only.
- Three seeds per configuration. The ranges are min and max, not confidence intervals. Differences smaller than the
  ranges, such as the three trainable sets, should be read as no difference.
- Wall times in the results files were taken on a shared, heavily loaded machine and are not benchmarks.
- No GPU runs. Nothing here needs one.
- C9 runs visual GRPO on a real pretrained VLM (SmolVLM-256M) with a freeze ablation and a weak-reward hack. This lab
  does not repeat that.

