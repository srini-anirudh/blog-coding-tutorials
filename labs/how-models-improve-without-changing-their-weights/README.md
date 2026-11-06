# Lab: how models improve without changing their weights

This lab is the code companion to the post
[How Models Improve Without Changing Their Weights](https://srini-anirudh.github.io/blogs/how-models-improve-without-changing-their-weights/).
The post argues that test-time compute is a fourth scaling knob. Generation and selection scale differently, and
the open problem is how to allocate the budget. The lab makes each step of that argument happen in small Python
code you can read.

The weights never change. Every experiment either simulates a sampler or queries a frozen model.

What you can run:

1. **The coverage law.** Each problem's coverage is a sigmoid. A population with Beta-distributed difficulty turns
   them into a power law with slope -a. A hopeless fraction puts a floor under it.
2. **The majority-vote ceiling.** Majority voting converges to "is the modal answer correct". Coverage converges to
   1 - pi0. The gap between them is what a better selector could still recover.
3. **The false-positive ceiling.** Picking a random passer gives accuracy p / (p + (1-p) phi), whatever k is. The
   lab derives the exact finite-k formula and checks it against Monte Carlo.
4. **Wide against deep.** K trajectories of T tokens against one of K*T tokens. The lab gives the analytic bill and
   wall-clock timings on a real model.
5. **Real repeated sampling.** A frozen Qwen2.5-0.5B-Instruct is sampled 64 times per problem on a graded arithmetic
   set. On those samples the lab measures coverage, majority voting, best-of-k by the model's own log-prob, and two
   imperfect verifiers.
6. **Budget forcing.** s1-style forced caps and "Wait" extensions on the same frozen model. The lab measures
   control, accuracy and right-to-wrong flips.
7. **The inverted U.** A three-state walk shows why extra steps can lower accuracy, and why easy problems peak first.
8. **Water-filling.** Greedy allocation of a sample budget, the closed form b* ~ (1/p) ln(p / lambda) and the peak
   at p = e lambda. The lab then compares allocation policies on the real success rates.

## Layout

```
labs/how-models-improve-without-changing-their-weights/
├── ttc/
│   ├── coverage.py     # per-problem coverage, Beta mixture closed form, pass@k, beta-binomial fit
│   ├── selection.py    # majority vote, false-positive ceiling, exact verifier accuracy, selector curves
│   ├── cost.py         # FLOP, decode-step and KV-cache bill for wide against deep
│   ├── walk.py         # the three-state reasoning walk
│   ├── allocate.py     # greedy water-filling, closed form, stop-when-solved
│   ├── arith.py        # seeded arithmetic benchmark, answer parser, two imperfect verifiers
│   ├── lm.py           # batched KV-cache decoding: temperature sampling and budget forcing
│   └── report.py       # results files that start with the hardware line and the command
├── scripts/            # 01 to 09, listed below
├── tests/              # pytest; a scripted fake model tests the decoding loops offline
└── results/            # real outputs of every script
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Scripts 04, 05 and 07 download small public models from
the Hugging Face Hub (SmolLM2-135M-Instruct and Qwen2.5-0.5B-Instruct). Nothing else is downloaded: the arithmetic
benchmark is generated from a seed.

## Run everything

```bash
L=labs/how-models-improve-without-changing-their-weights

# Tests. The network test downloads a tiny random GPT-2.
python -m pytest $L -m "not slow and not gpu and not network"
python -m pytest $L -m network

# Simulations and closed forms. Seconds each on a CPU.
python $L/scripts/01_coverage_law.py
python $L/scripts/02_majority_ceiling.py
python $L/scripts/03_false_positive_ceiling.py
python $L/scripts/08_inverted_u.py

# Wide against deep: the analytic bill and timings. The deep runs are slow: the CPU run took about 40 minutes on a loaded
# machine, the GPU run about 6 minutes.
python $L/scripts/04_width_vs_depth.py --shapes 8x128,32x64
python $L/scripts/04_width_vs_depth.py --shapes 8x128,32x64,128x32 --repeats 3   # on a GPU

# Real samples from a frozen model. Use a GPU: 96 problems x 64 samples.
python $L/scripts/05_sample_real.py --rows 1024
python $L/scripts/05_sample_real.py --problems 16 --k 8          # CPU smoke run, writes the same file
python $L/scripts/06_real_scaling.py                               # CPU, reads results/05_samples.json
python $L/scripts/09_allocation.py                                 # CPU; part 2 reads 05_samples.json

# Budget forcing on the same model. Use a GPU.
python $L/scripts/07_budget_forcing.py
```

Which script covers which section of the post:

| Post section | Script |
|---|---|
| Eight ways to spend a budget | all of them; this lab covers repeated sampling, voting, best-of-N, verification and budget forcing |
| What the extra tokens cost | `04_width_vs_depth.py` |
| Why sampling has a scaling law (Figure 2) | `01_coverage_law.py`, `06_real_scaling.py` |
| The generator-verifier gap (Figure 3) | `02_majority_ceiling.py`, `06_real_scaling.py` |
| The false-positive ceiling (Figure 4) | `03_false_positive_ceiling.py`, `06_real_scaling.py` |
| Nuance 07: curves are benchmark-shaped | `06_real_scaling.py` (stratified by level) |
| Budget forcing (Figure 5), Nuance 08 | `07_budget_forcing.py` |
| When more thinking hurts (Figure 6) | `08_inverted_u.py`, flips in `07_budget_forcing.py` |
| Allocate, don't extend (Figure 7), Nuance 09 | `09_allocation.py` |

## Results

Every number names its hardware. Raw output is in `results/`; each file starts with a hardware line and the command.
Scripts 01, 02, 03, 08, 09 and 06 ran on AMD EPYC 7R13, 8 threads, torch 2.15.0 (CPU, seeded). Scripts 05, 07 and the
GPU part of 04 ran on NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2. Models: Qwen2.5-0.5B-Instruct
(scripts 05, 06, 07, 09) and SmolLM2-135M-Instruct (script 04).

Tests on AMD EPYC 7R13, 8 threads: 25 passed offline (1 `network` test deselected); the `network` test passes on its own.
There are no `slow` or `gpu` tests.

### Simulations and closed forms (AMD EPYC 7R13, 8 threads)

| Measurement | Result |
|---|---|
| One problem, p = 0.01: coverage at k = 10 / 100 / 1000 | 0.096 / 0.634 / 1.000 (a sigmoid, half-way at k = 69) |
| Beta(a, 3) population: fitted log-log slope of failure fraction, k >= 30, a = 0.35 / 0.70 / 1.40 | -0.347 / -0.699 / -1.231 (Monte Carlo, 20,000 problems); exact formula -0.346 / -0.691 / -1.379 |
| Same, with 12% hopeless problems, a = 0.70: failure fraction at k = 10,000 | 0.123 (floor at pi0 = 0.12) |
| Synthetic answer distributions, 300 problems, 256 samples: coverage / majority at k = 256 | 0.887 / 0.484 |
| Same: coverage limit / majority limit / gap | 0.917 / 0.473 / 0.443 |
| False-positive ceiling at p = 0.05, phi = 0 / 0.01 / 0.05 / 0.20 | 1.000 / 0.840 / 0.513 / 0.208 |
| Exact verifier formula against Monte Carlo (20,000 trials), largest gap over 12 (p, phi) pairs and 6 values of k | about 0.01 |
| Reasoning walk, easy problem, moderate pull: peak step / peak accuracy / accuracy at step 200 | 28 / 0.968 / 0.752 |
| Same, hard problem | 69 / 0.670 / 0.546 |
| Same, no pull | monotone, 1.000 (easy) and 0.786 (hard) at step 200 |
| Water-filling, 12 problems, B = 150: expected solved, uniform / water-filling | 5.94 / 6.71 |
| Same, B = 400 | 7.59 / 8.36 |

The Monte Carlo slope for a = 1.40 is noisy because fewer than 10 of 20,000 problems are unsolved at large k.
The exact formula is the reference.

### Real samples from a frozen model (96 problems, 64 samples, temperature 0.7)

Samples: NVIDIA H100 80GB HBM3, 966,859 tokens in 254 s. Analysis: AMD EPYC 7R13, 8 threads.

| Measurement | Result |
|---|---|
| Mean per-sample accuracy / problems never solved in 64 samples | 0.499 / 7 of 96 |
| Coverage (pass@k) at k = 1 / 4 / 16 / 64 | 0.499 / 0.756 / 0.873 / 0.927 |
| Majority vote at k = 1 / 4 / 16 / 64 | 0.501 / 0.627 / 0.776 / 0.822 |
| Best-of-k by the model's mean token log-prob at k = 1 / 4 / 16 / 64 | 0.501 / 0.575 / 0.599 / 0.557 |
| Verifier "last digit matches": false-positive rate / accuracy at k = 64 / formula | 0.491 / 0.621 / 0.623 |
| Verifier "last digit and length match": false-positive rate / accuracy at k = 64 / formula | 0.367 / 0.666 / 0.666 |
| Fitted Beta(a, b) to the success counts | a = 0.697, b = 0.794 |
| Log-log slope of failure fraction, k >= 4: measured / fitted -a | -0.422 / -0.697 |
| Easy levels (first four) against hard levels (last four): failure at k = 64 | 0.000 / 0.146 |
| Held-out half, expected solved of 48 at 4 samples per problem: uniform / oracle water-fill / level-predicted / stop-when-solved | 36.85 / 39.28 / 36.12 / 43.09 |

What this shows. Coverage keeps rising to 0.927 at k = 64. Majority voting reaches 0.822 and flattens. The model's own
log-prob peaks near k = 16 and then falls, so it is a poor selector for this task. The exact verifier formula
reproduces the measured accuracy of the two partial verifiers to within 0.01 at k >= 16. The Beta fit predicts a
steeper tail than the data show (slope -0.70 against -0.42), so the fit is only a rough guide at k <= 64, in line with
Nuance 03. The benchmark's mix of levels sets the curve: the easy levels reach failure 0 and the hard levels do not.

The "level-predicted" allocation uses the mean success rate of each difficulty level, measured on the other half of the
problems. It does not beat uniform at 4 samples per problem. The stop-when-solved policy wins because it has a perfect
verifier, which the other policies are not given. The per-problem success rates in this comparison are measured
rates treated as truth, so the comparison is a model of allocation, not a rerun of the model.

### Budget forcing (96 arithmetic problems, greedy, 1024 thinking-token limit; NVIDIA H100 80GB HBM3)

| Setting | Thinking tokens (mean) | Accuracy | Right to wrong | Wrong to right |
|---|---|---|---|---|
| Natural stop | 178.7 | 0.500 | 0 | 0 |
| Cap 16 / 32 / 64 / 128 | 16.0 / 32.0 / 64.0 / 118.5 | 0.375 / 0.375 / 0.198 / 0.365 | 27 / 25 / 39 / 14 | 15 / 13 / 10 / 1 |
| "Wait" x1 / x2 / x4 / x8 | 273.3 / 533.0 / 687.6 / 1024.0 | 0.479 / 0.479 / 0.500 / 0.479 | 2 / 5 / 2 / 5 | 0 / 3 / 2 / 3 |
| "Hmm" x2 | 363.2 | 0.458 | 7 | 3 |

Control works: the knob moves the token count. Scaling does not, for this model: no "Wait" setting beats the natural
stop, and at x8 every trace hits the 1024-token limit. Qwen2.5-0.5B-Instruct was not fine-tuned on reasoning traces, so
this is not the s1 setup (a 32B model fine-tuned on 1,000 traces). The paper's gains were not reproduced and are not
claimed. The counting example in the results file shows the failure the post describes in Nuance 08: after repeated
"Wait" the trace loops on one paragraph. "Hmm" does no better than "Wait" here, so the lab does not support the claim that
"Wait" is special. With 96 problems, one problem is 0.010, so differences of a few problems are noise.

### Wide against deep (SmolLM2-135M-Instruct, temperature-1 sampling, KV cache, no early stop)

The analytic bill matches the post's table: weight FLOPs and KV slots are equal, attention FLOPs grow by about K and
serial decode steps by exactly K. Measured wall clock tracks the step ratio, not the attention ratio.

| Shape (K x T) | Deep/wide attention FLOPs | Deep/wide decode steps | Deep/wide wall clock, NVIDIA H100 80GB HBM3 | Deep/wide wall clock, AMD EPYC 7R13, 8 threads |
|---|---|---|---|---|
| 8 x 128 | 7.95 | 8 | 12.18 | 12.71 |
| 32 x 64 | 31.52 | 32 | 32.25 | 11.53 |
| 128 x 32 | 124.15 | 128 | 100.61 | not run |

On the GPU the model is small, so each step is dominated by kernel launches and the Python loop. The wide run is fast
because it does K tokens per step. This is a statement about step count and latency, not about attention cost. The CPU run
shared a loaded machine, and its 32 x 64 ratio is well below the step ratio. Treat all timings as orders of magnitude.

## What was not measured

- The large-scale results the post quotes were not reproduced: Large Language Monkeys coverage on SWE-bench Lite,
  the 14x-larger-model comparison of Snell et al., REBASE against MCTS in Wu et al., s1's 32B results, and the
  inverse-scaling results of Gema et al. The lab shows the mechanism at small scale only.
- No process reward model, tree search or iterative refinement is built. Search appears only as the idea that a
  verifier at intermediate states is needed.
- The inverted U comes from a toy walk with chosen parameters. It is not a measurement of any language model.
- Model-based verifiers and self-verification are not tested. The two verifiers are cheap rules on the answer.
- The attention-FLOP asymmetry is computed, not measured; the wall-clock timings are launch-bound at this model size.
- No temperature sweep, and one seed for the real samples.

