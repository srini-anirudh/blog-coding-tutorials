# Lab: Mixing Data Without Losing Capabilities

Runnable code for the theory post [Mixing Data Without Losing Capabilities](https://srini-anirudh.github.io/blogs/mixing-data-without-losing-capabilities/).

The post argues that a data mixture is a loss weighting. Every mixing decision is a choice of coefficients in a weighted
sum of objectives that interfere. This lab makes that idea something you can run. It trains a small transformer on
synthetic tasks with exact answers, and it measures the mechanisms the post describes:

- the gradient Gram matrix K, and the first-order law ΔL ≈ −ηKp;
- how a token-mean loss turns a 30/30/40 example mixture into a different gradient mixture;
- what proportional, equal, temperature and capped mixing do to 24 sources of very different sizes;
- forgetting under sequential tuning, against transfer under a joint mixture;
- leave-one-out transfer matrices, a fitted mixture law, and four objectives checked against stratified sampling;
- replay as retention bought with adaptation;
- the loss-chasing trap in online mixing;
- the four-rung diagnostic ladder for a dropped metric.

Everything runs on a CPU in a few minutes per script. Nothing here reproduces a paper's numbers at scale. Where the post
quotes large-scale results (Tülu 3, Aioli, Dong et al., Ibrahim et al., spurious forgetting), the lab shows the
mechanism small and says so.

## The setup

The model is the Llama-style `TinyLLM` built in [C4](../../c04-llm/). The lab imports it and only shrinks the
configuration: 2 layers, width 48, 4 heads, a 21-symbol vocabulary. That is 47,328 parameters. Script 04 also runs a
width-16 model with 5,536 parameters.

The tasks stand in for the post's sources. Each one has an exact answer, so a drop in a score is a real drop.

| Task | Example | Stands in for |
|---|---|---|
| `add` | `37+45=` → `082` | maths (the slowest skill to learn here) |
| `sub` | `45-37=` → `08` | a second maths source |
| `rev` | `R3745=` → `5473` | code |
| `sort` | `S3745=` → `3457` | code |
| `copy` | `C3745=` → `3745` | chat (learned fast, then saturates) |
| `noise` | `N3745=` → 4 random digits | a source with a large irreducible loss |
| `long` | `L3745=` → `374537453745` | long chain-of-thought answers |

Accuracy is exact match on 1,000 held-out examples per task. Under teacher forcing, every answer token must be the
argmax. That equals greedy decoding, because greedy decoding only departs from teacher forcing after its first wrong
token. Script 08 also uses free generation, because it needs to see the output format.

## Layout

```
labs/mixing-data-without-losing-capabilities/
├── mixlab/
│   ├── tasks.py        # the synthetic tasks, tokenisation, supervised-only labels, response styles
│   ├── model.py        # imports TinyLLM from c04-llm and shrinks it
│   ├── train.py        # mixture sampler, token-mean vs per-example loss, training loop, evaluation, greedy decoding
│   ├── grads.py        # per-task gradients, Gram matrix, PCGrad, one-step first-order check
│   ├── mixtures.py     # proportional / equal / temperature / capped rules, N_eff, epochs, token weights
│   ├── objectives.py   # mean, max-min, weighted, constrained objectives; mixture surrogate; simplex search
│   └── runs.py         # parallel runs over forked processes; results files with a hardware line
├── scripts/            # one script per mechanism, listed below
├── results/            # the real output of every script (.txt, plus .json with the raw numbers)
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. The lab needs only PyTorch and the standard library.

Scripts that train many models run them as separate processes with one thread each. `MIXLAB_PROCS` sets how many
(default 8). The values used for the results are in the commands below.

## Run everything

```bash
# Tests (about 10 s).
python -m pytest labs/mixing-data-without-losing-capabilities -m "not slow and not gpu and not network"

# Scripts. Wall times are from the results files (AMD EPYC 7R13, shared machine).
python labs/mixing-data-without-losing-capabilities/scripts/01_gram_matrix.py                       # ~2 min, 4 threads
MIXLAB_PROCS=9  python labs/mixing-data-without-losing-capabilities/scripts/02_token_weighting.py   # ~30 s
python labs/mixing-data-without-losing-capabilities/scripts/03_naive_mixtures.py                    # instant, no training
MIXLAB_PROCS=12 python labs/mixing-data-without-losing-capabilities/scripts/04_forgetting_vs_transfer.py              # ~4 min
MIXLAB_PROCS=12 python labs/mixing-data-without-losing-capabilities/scripts/04_forgetting_vs_transfer.py --hidden 16  # ~4 min
MIXLAB_PROCS=20 python labs/mixing-data-without-losing-capabilities/scripts/05_transfer_and_mixture_law.py            # ~5 min
MIXLAB_PROCS=12 python labs/mixing-data-without-losing-capabilities/scripts/06_replay.py            # ~4 min
MIXLAB_PROCS=6  python labs/mixing-data-without-losing-capabilities/scripts/07_loss_chasing.py      # ~3 min
MIXLAB_PROCS=8  python labs/mixing-data-without-losing-capabilities/scripts/08_diagnostic_ladder.py # ~19 min
```

## Which script covers which part of the post

| Post section | Script |
|---|---|
| §1 The whack-a-mole problem (figure 1) | `04_forgetting_vs_transfer.py` |
| §2 Negative transfer vs catastrophic forgetting, the diagnostic | `04_forgetting_vs_transfer.py` |
| §2 A third thing that is neither (spurious forgetting) | `08_diagnostic_ladder.py` |
| §3 Why tasks interfere: ΔL ≈ −ηKp, the Gram matrix, PCGrad | `01_gram_matrix.py` |
| §4 A mixture is a loss weighting; tokens, not examples | `02_token_weighting.py` |
| §5 The naive mixtures, temperature, capping, the counting question | `03_naive_mixtures.py` |
| §7 What are we optimising: mean, max-min, weighted, constrained; non-convex fronts | `05_transfer_and_mixture_law.py` (parts 0, 5) |
| §7 Your objective is only defined up to evaluation noise | `05_transfer_and_mixture_law.py` (part 1) |
| §8 The transfer matrix, leave-one-out, sign changes with budget | `05_transfer_and_mixture_law.py` (part 2) |
| §9 Mixture laws, fit then verify, the stratified baseline | `05_transfer_and_mixture_law.py` (parts 3-5) |
| §10 The loss-chasing trap | `07_loss_chasing.py` |
| §11 The diagnostic ladder | `08_diagnostic_ladder.py` |
| §12 Replay | `06_replay.py` |

## Results

Every number below came from a script in this folder, on **AMD EPYC 7R13, CPU only, torch 2.15.0**. The machine was
shared with other jobs. That changes wall times, not the numbers. Each results file starts with its hardware line and
its command. Accuracies are exact match in percent. "2 seeds" means the mean of two training seeds.

### §3 The Gram matrix and the first-order law (`01_gram_matrix.txt`)

After 300 steps of a uniform five-task mixture:

| Measurement | Result |
|---|---|
| Cosine between task gradients | +0.05 to +0.21 for nine of the ten pairs; `add` vs `copy` −0.07 |
| One SGD step along the uniform mixture, η = 0.001: actual vs predicted ΔL | relative error 0.02 |
| Same, η = 0.01 / 0.1 / 1.0 | relative error 0.15 / 1.05 / 4.49 |
| One step on `add` alone, η = 0.001: ΔL on `copy` | +0.0001 actual, +0.0001 predicted (the conflict is real) |
| PCGrad on `add` vs `copy`, η = 0.03: ΔL on `copy` | +0.0059 → +0.0011, while ΔL on `add` stays −0.030 |

After 1,500 steps, most cosines shrink to within ±0.05. The most conflicting pair becomes `add` vs `sort` (−0.08). A
step on `add` raises the `sort` loss by +0.0127; with PCGrad, by +0.0033. PCGrad does not reach zero, because the
step size brings in the second-order term. At η = 0.1 the summed loss already rises. That is the curvature term the
post warns about.

### §4 Tokens, not examples (`02_token_weighting.txt`)

Sources: `long` (13 supervised tokens), `rev` and `copy` (5 each), sampled 30/30/40 by example.

| Measurement | Result |
|---|---|
| Gradient share of `long` under token-mean loss: formula / measured over 2,000 batches | 0.527 / 0.526 |
| Gradient share of `rev`, `copy` | 0.203, 0.270 (nominal 0.30, 0.40) |
| Example shares that restore a 30/30/40 token share | 0.142 / 0.368 / 0.491 |
| Exact identity ∇(token mean) = Σ (Tᵢ/T) ∇Lᵢ on one batch | max difference 3.0e-8 |
| After 400 steps, `rev` accuracy: token-mean / per-example / token-mean with corrected p (3 seeds) | 84.8 / 98.2 / 96.5 |

A 30% share of long answers became 53% of the gradient. The short sources paid for it.

### §5 Naive mixtures on 24 sources (`03_naive_mixtures.txt`)

Sizes from 700 to 2,000,000 examples, log-spaced. Budget one million examples.

| Rule | N_eff | Largest share | Epochs on the 700-example source |
|---|---|---|---|
| proportional | 7.9 | 29.3% | 0.15 |
| temperature α = 0.7 | 11.0 | 21.6% | 1.17 |
| temperature α = 0.5 | 14.5 | 16.1% | 4.31 |
| temperature α = 0.3 | 19.2 | 10.7% | 14.1 |
| equal | 24.0 | 4.2% | 59.5 |
| capped κ = 10,000 | 20.9 | 5.4% | 5.40 |
| capped κ = 50,000 | 17.0 | 7.1% | 1.43 |

The post's "700-example task in a 24-task equal mixture is trained for roughly sixty epochs" is 59.5 epochs. One
qualification: under a fixed budget, capping gives every source below the cap the same number of epochs. A cap of 10,000
gives the smallest source 5.4 epochs. So capping does not up-sample relative to the other small sources, but it can still
repeat them.

### §1-2 Forgetting vs transfer (`04_forgetting_vs_transfer.txt`, `04_forgetting_vs_transfer_hidden16.txt`)

Every task gets 1,500 steps of data in every setting. 2 seeds.

| Setting | copy | add | rev | sort | mean |
|---|---|---|---|---|---|
| width 48: specialists | 100.0 | 93.2 | 100.0 | 98.7 | 98.0 |
| width 48: sequential copy → add → rev → sort, at the end | 7.3 | 0.0 | 5.9 | 98.5 | 27.9 |
| width 48: joint mixture, 6,000 steps | 100.0 | 95.8 | 99.9 | 96.8 | 98.1 |
| width 16: specialists | 100.0 | 36.5 | 100.0 | 99.0 | 83.9 |
| width 16: sequential, at the end | 7.5 | 0.0 | 7.2 | 99.7 | 28.6 |
| width 16: joint mixture, 6,000 steps | 99.5 | 77.4 | 99.2 | 94.1 | 92.5 |

Sequential tuning erases each earlier skill almost completely. The model is trained from scratch, so it has no
pretrained features to fall back on. That makes forgetting harsher here than in an LLM.

The joint mixture keeps everything. At width 48 it matches the specialists. At width 16 it shows both signs of transfer:
`add` is 40.9 points above its specialist (positive transfer), `sort` is 4.9 points below (negative transfer). The post's
figure 1 shows specialists above the mixture on every skill. This lab reproduces that gap only for `sort`.

### §7-9 Objectives, transfer matrix, mixture law (`05_transfer_and_mixture_law.txt`)

Four sources (`add`, `rev`, `sort`, `copy`), 1,500 steps per run.

| Measurement | Result |
|---|---|
| Stratified (uniform), 3 seeds: `add` accuracy | 7.2, 90.1, 39.1 (seed std 41.8 points) |
| Seed std of `rev`, `sort`, `copy` | 0.3, 2.2, 0.3 |
| Leave-one-out, effect of including `sort` on `add`: 1,500 / 4,500 steps (2 seeds) | +18.9 / −3.3 points |
| Leave-one-out, effect of including `copy` on `add`: 1,500 / 4,500 steps | +32.7 / +6.3 points |
| Surrogate held-out mean error (4 random mixtures) | 7.4 points |

Objective → mixture picked on the fitted surface → real runs (2 seeds):

| Objective | Mixture (add, rev, sort, copy) | add | rev | sort | copy | mean | min |
|---|---|---|---|---|---|---|---|
| stratified (3 seeds) | 0.25 / 0.25 / 0.25 / 0.25 | 45.5 | 99.8 | 94.8 | 99.8 | 85.0 | 45.5 |
| mean | 0.55 / 0.15 / 0.15 / 0.15 | 81.5 | 89.9 | 78.4 | 98.8 | 87.2 | 78.4 |
| max-min | 0.50 / 0.15 / 0.20 / 0.15 | 69.5 | 89.3 | 84.2 | 98.8 | 85.5 | 69.5 |
| weighted u = (.4, .2, .2, .2) | 0.55 / 0.15 / 0.15 / 0.15 | 81.5 | 89.9 | 78.4 | 98.8 | 87.2 | 78.4 |
| max `add` s.t. others ≥ stratified − 2 | 0.25 / 0.25 / 0.30 / 0.20 | 31.2 | 99.2 | 94.9 | 99.8 | 81.3 | 31.2 |

How to read this:

- The noise floor decides almost everything. `add` sits near a learning threshold at 1,500 steps, so its seed
  standard deviation is 41.8 points. Every difference in the `add` column is inside that band. The same is true of the
  leave-one-out sign change for `sort` (+18.9 at 1,500 steps, −3.3 at 4,500). It looks like the budget-dependent sign
  flip in the post, but one cannot tell it from noise with 2 seeds.
- The fitted mixtures raise the weakest skill above stratified (min 78.4 and 69.5 against 45.5). They pay with `rev`
  and `sort`, which drop 10 to 16 points. The mean moves by 0.5 to 2.2 points, well inside noise. In the post's terms,
  this run does not beat stratified sampling on the mean.
- In an earlier version of this script (its output is not kept in `results/`), the surrogate was not clipped to
  [0, 100]. The weighted objective then extrapolated `add` to a predicted 176% and picked 85% `add`. The real runs
  scored 6.2 on `rev`. Clipping the prediction to the range accuracy can take fixed that pick. It is a concrete case of the post's point that a fitted law is a proposal mechanism, not an acceptance test.
- Part 0 checks the post's Model A / Model B example (means 74.75 and 75.0, minimums 15 and 71). It also shows that on a
  front with a dent, no weighted sum picks the balanced point (0.55, 0.55), while max-min does.

### §12 Replay (`06_replay.txt`)

Stage 1: `add` for 3,000 steps (97.5%). Stage 2: 1,500 steps of a new task with a fraction λ replayed from `add`. 2 seeds.

| Replay λ | 0% | 1% | 5% | 10% | 25% | 50% |
|---|---|---|---|---|---|---|
| new = `sub`: retention of `add` | 0.0 | 1.5 | 57.3 | 67.2 | 97.2 | 95.8 |
| new = `sub`: acquisition | 100.0 | 97.5 | 99.5 | 98.8 | 99.2 | 98.8 |
| new = `rev`: retention of `add` | 0.0 | 2.0 | 32.9 | 51.6 | 89.8 | 96.1 |
| new = `rev`: acquisition | 100.0 | 99.9 | 99.5 | 99.8 | 99.5 | 97.8 |

Without replay, `add` is gone after either new task. The unrelated task (`rev`) needs more replay than the related one
(`sub`) at every ratio from 5% to 25%. The max-min optimum is 25% for `sub` and 50% for `rev`. That matches the
direction of Ibrahim et al.'s finding that a stronger shift needs more replay. Acquisition barely moves, because these
tasks are easy for the model. The cost of replay shows up as at most 2.5 points of acquisition here.

### §10 The loss-chasing trap (`07_loss_chasing.txt`)

Five sources, one of them pure noise. 3,000 steps. 2 seeds.

| Schedule | Budget to `noise` | add | rev | sort | copy | mean | min |
|---|---|---|---|---|---|---|---|
| static uniform | 20.0% | 85.3 | 100.0 | 97.6 | 100.0 | 95.7 | 85.3 |
| up-weight the highest loss | 56.9% | 34.5 | 97.2 | 87.1 | 97.8 | 79.1 | 34.5 |
| up-weight the fastest loss drop | 15.1% | 11.9 | 93.7 | 87.1 | 99.2 | 72.9 | 11.9 |

The loss-level controller gives the noise source up to 69% of each batch. The noise loss never falls below 1.84 nats
per token, so the controller keeps feeding it. `add` loses 51 points.

The derivative controller avoids the trap: it gives noise less budget than static mixing. It is still worse than static.
It moves weight greedily to whichever source improved most in the last 100 steps. It put about 85% on `add` early, then
dropped `add` to 2% at step 1,500 while `add` was still unfinished. This is the "noisy controller chases short-term
gains" failure from the post's comparison table. The post's own conclusion, that a static stratified baseline is hard
to beat, is what this run shows.

### §11 The diagnostic ladder (`08_diagnostic_ladder.txt`)

Stage 1: `add` for 3,000 steps (strict 97.4%). Stage 2: only `rev` and `sort`, in the plain style or with a `>`
preamble on every answer. Then `add` is measured on each rung. One seed. Rung 4 is the number of `add` training steps
needed to get back to 95.4% (stage-1 score minus 2). Learning `add` from scratch took 3,000 steps.

| Stage-2 style | Stage-2 steps | `add` outputs starting with `>` | Strict | Rung 1: lenient parser | Rung 2: 100 examples, 40 steps | Rung 3: ranking | Rung 4: steps to recover |
|---|---|---|---|---|---|---|---|
| plain | 30 | 0% | 4.2 | 4.2 | 89.8 | 15.0 | 25 |
| plain | 100 | 0% | 0.6 | 0.6 | 29.8 | 1.0 | 100 |
| plain | 300 | 0% | 0.2 | 0.2 | 5.8 | 2.0 | 300 |
| plain | 3,000 | 0% | 0.0 | 0.0 | 2.6 | 3.5 | 675 |
| preamble | 30 | 0% | 5.4 | 5.4 | 82.4 | 10.0 | 25 |
| preamble | 100 | 80% | 0.0 | 0.0 | 25.2 | 2.5 | 125 |
| preamble | 300 | 92% | 0.2 | 0.2 | 15.4 | 1.0 | 175 |
| preamble | 3,000 | 84% | 0.0 | 0.0 | 3.2 | 4.5 | 350 |

What this shows, and what it does not:

- The preamble does leak. After 100 steps, 80% of `add` answers start with `>`, a convention the model never saw on
  `add`. That is the style transfer the post describes in §2 and §12.
- But rung 1 never helps here. Stripping the `>` restores nothing, because the digits after it are also wrong. In this
  tiny model trained from scratch, the format change and real damage arrive together. The post's spurious-forgetting
  result (a drop that is mostly a readout change) is **not** reproduced.
- Rungs 2 and 4 show the post's real distinction: displaced versus destroyed. After 30 steps of new data, `add` reads
  near 5%, yet 100 examples bring it back to 82-90%, and 25 steps of `add` restore it fully. After 3,000 steps, the
  same 100 examples give 3%, and recovery takes 350-675 steps. That is still 4-9 times faster than learning from
  scratch, so even then something survived.
- Rung 3, ranking the true sum against 9 near misses, stays at 1-15%. It does not detect the knowledge that rung 2
  shows is still cheaply recoverable after 30 steps. A probe this crude can miss what is there.

## What was not measured

- No result in the post at LLM scale was reproduced. That includes Tülu 3's ablations, Dong et al.'s saturation curves,
  Aioli against stratified sampling, the 0.66% SFT mixture-law gap, Biderman et al.'s LoRA forgetting numbers, the
  Bianchi et al. 3% safety result, and the spurious-forgetting paper's experiments. The lab shows the mechanisms on
  tiny tasks only.
- The model is trained from scratch, not fine-tuned from a pretrained base. Forgetting is therefore harsher than in
  post-training, and representations have little to survive on.
- Regularisation (L2-to-init, EWC), LoRA, model merging, MoE routing, decontamination, deduplication and the safety
  "hump" have no script. The post discusses them. They are left out to keep the lab focused.
- Most training results use 2 seeds, and the stratified baseline uses 3. The `add` capability is very seed-sensitive
  at 1,500 steps, so read small gaps as ties.
- No GPU was used. Nothing in the lab needs one.

## Tests

`python -m pytest labs/mixing-data-without-losing-capabilities -m "not slow and not gpu and not network"`: 18 passed on
AMD EPYC 7R13 (about 5 s). They cover the task labels, the sampler, the token-mean gradient identity, the
first-order step check, PCGrad, the mixing rules, the 59.5-epoch example, the objectives and the surrogate. There are
no `slow`, `gpu` or `network` tests.
