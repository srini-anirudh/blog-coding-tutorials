# Lab: Reinforcement learning from first principles

Code companion to the theory post
[Reinforcement Learning from First Principles](https://srini-anirudh.github.io/blogs/reinforcement-learning-from-first-principles/).

The post derives the policy gradient, baselines, advantages, GAE and PPO's clip. Then it explains why each piece
bends for language models. This lab checks those claims on a policy small enough to list every answer.

The policy is a tiny autoregressive model, `TinyLM`. It reads a prompt token and emits 5 tokens from 6 digits. That
is 7,776 possible completions per prompt. Because they can all be listed, the quantities the post calls
intractable become exact sums: J, the true gradient, the values V(s_t), the advantages, the entropy, the KL between
two policies and pass@k. Every sampled estimator in the lab is measured against that ground truth.

The last script leaves the toy. It scores the same SmolLM2-135M samples along two paths, a sampler path and a
trainer path, to show the "on-policy illusion" from section 14.

C6 builds REINFORCE and GRPO on a real chat model, and C12 builds critic-based advantages for agents. This lab does
not repeat them. It measures the claims those tutorials take as given.

## Layout

```
labs/reinforcement-learning-from-first-principles/
├── rlfp/
│   ├── policy.py       # TinyLM: state = prefix, action = token; sampling, log-probs, values
│   ├── exact.py        # list every completion: exact J, gradient, V(s_t), A, entropy, KL, pass@k
│   ├── tasks.py        # rewards: digit sum, key token, exact match
│   ├── estimators.py   # REINFORCE loss, baselines (batch, leave-one-out, group, group-std), error stats
│   ├── advantages.py   # returns, TD errors, GAE with a mask
│   ├── ppo.py          # clipped surrogate, the four regions, entropy, KL estimators k1 k2 k3
│   ├── train.py        # one loop, six advantage estimators
│   ├── tensors.py      # masks, loss normalisation, chunked log-probs, memory accounting
│   ├── numerics.py     # sampler path vs trainer path log-probs, and their mismatch statistics
│   └── report.py       # results files with a hardware line and the command
├── scripts/            # 01 to 10, one per group of post sections
├── tests/              # pytest
└── results/            # outputs written by the scripts
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. PyTorch uses every core by default, so set
`OMP_NUM_THREADS` on a shared machine. The numbers below used `OMP_NUM_THREADS=8`.

## Run everything

```bash
python -m pytest labs/reinforcement-learning-from-first-principles -m "not slow and not gpu and not network"
python -m pytest labs/reinforcement-learning-from-first-principles -m network   # downloads SmolLM2-135M

python labs/reinforcement-learning-from-first-principles/scripts/01_policy_gradient.py
python labs/reinforcement-learning-from-first-principles/scripts/02_variance.py
python labs/reinforcement-learning-from-first-principles/scripts/03_baselines.py
python labs/reinforcement-learning-from-first-principles/scripts/04_advantages_gae.py
python labs/reinforcement-learning-from-first-principles/scripts/05_trust_region.py
python labs/reinforcement-learning-from-first-principles/scripts/06_ppo_clip.py
python labs/reinforcement-learning-from-first-principles/scripts/07_compare_estimators.py
python labs/reinforcement-learning-from-first-principles/scripts/08_rl_step_shapes.py
python labs/reinforcement-learning-from-first-principles/scripts/09_cold_start.py
python labs/reinforcement-learning-from-first-principles/scripts/10_on_policy_illusion.py           # SmolLM2-135M
python labs/reinforcement-learning-from-first-principles/scripts/10_on_policy_illusion.py --random  # no download
```

Each script writes `results/<script>.txt` and `results/<script>.json`. Every file starts with the hardware and the
command. Scripts 01, 08 and 09 take seconds. Scripts 02 to 07 take one to several minutes each on a free 8-thread
CPU; on a loaded machine they took longer. Script 10 takes a few minutes on CPU and seconds on a GPU.

## Which script covers which section

| Post section | Script | What it shows |
|---|---|---|
| A language model is already a policy | `rlfp/policy.py` | state = prefix, action = token, transition = append |
| The objective | `01_policy_gradient.py` | exact J; the greedy trap: same J, no gradient |
| The policy gradient, derived | `01_policy_gradient.py` | score-function form vs exact gradient and finite differences; error x sqrt(N) is flat |
| Why the estimator is so noisy | `02_variance.py` | share of zero-reward samples and of all-equal groups; SNR against sequence length |
| Baselines, and why they are free | `03_baselines.py` | E[b grad log pi] = 0; the exact variance parabola; self-inclusion bias; per-state baseline |
| Values, advantages, actor-critic | `04_advantages_gae.py` | exact V, Q, A; where the credit lands |
| TD, bootstrapping and GAE | `04_advantages_gae.py` | GAE(1, 1) = R - V; the lambda dial with redrawn and persistent critic error |
| Why you cannot take a big step | `05_trust_region.py` | KL for a fixed parameter step; entropy collapse at four learning rates |
| PPO, which now feels inevitable | `06_ppo_clip.py` | the four regions; reusing one batch with and without the clip |
| Back to tensors | `08_rl_step_shapes.py` | every tensor shape of one step; the post-EOS mask bug; 20x token weights; 268 GB of logits |
| The four models in the room | `08_rl_step_shapes.py` | weights and optimiser state at 7B |
| Why RL lives in post-training | `09_cold_start.py` | exactly zero gradient from scratch; a short SFT stage fixes it; 1 - (1 - p)^G |
| Where the textbook bends, 1-3 | `07_compare_estimators.py`, `06_ppo_clip.py` | group baselines vs a learned critic; how often the clip fires |
| Where the textbook bends, 6 | `10_on_policy_illusion.py` | sampler vs trainer log-prob gaps in three dtypes, summed over a sequence |

## Results

Every number below is from the files in `results/`. Unless a row says otherwise, the hardware is an AMD EPYC 7R13
with 8 threads and torch 2.15.0. The toy runs are seeded, so their numbers do not depend on machine load. The task
is "6 digits, 5 tokens, reward 1 if the digits add up to the target". The targets are 4, 8, 12, 16 and 20. At the
start, about 1% of completions are correct for targets 4 and 20, and about 10% for target 12.

Tests (same CPU): 20 offline tests passed, and 1 `network` test passed. There are no `slow` or `gpu` tests.

### The gradient and its noise (scripts 01 to 03)

| Measurement | Result |
|---|---|
| Finite difference of the exact J along a random direction vs the score-function gradient (float64) | 1.253485e-04 vs 1.253485e-04 |
| REINFORCE RMS error relative to the true gradient, times sqrt(N), at N = 4 / 64 / 1024 | 26.2 / 25.8 / 24.9 (flat, so error falls like 1/sqrt(N)) |
| Share of N = 1 estimates that are exactly zero (target 12, pass rate 0.096) | 0.93 |
| Greedy trap: two policies with J = 0.6. Gradient norm, diverse vs collapsed | 0.082 vs 1.1e-13 |
| Groups of 8 where all rewards are equal, measured vs p^G + (1-p)^G, target 4 / 12 | 0.92 vs 0.92 / 0.44 vs 0.44 |
| SNR when only token 0 matters, T = 2 / 8 / 32 / 128 | 1.25 / 0.53 / 0.24 / 0.12. Log-log slope -0.57 (the post predicts -0.5) |
| E[grad log pi] summed over all 7,776 completions, vs the gradient norm | 1.0e-06 vs 8.0e-02 |
| Exact one-sample variance at b = 0 / E[R] / best b (4 digits x 4 tokens) | 3.24 / 2.71 / 2.70. Best b = 0.164, E[R] = 0.175 |
| Mean estimate projected on the true gradient, N = 4: batch mean incl. self / leave-one-out | 0.78 +- 0.04 / 1.04 +- 0.05 (predicted: 0.75 / 1) |
| Gradient MSE vs no baseline, N = 4: batch mean / leave-one-out / exact V(s_t) | 0.71 / 1.26 / 0.89 |

The best constant baseline sits close to the mean reward, as the post says. The last row is a caveat. The batch mean
has the lowest error at N = 4 because it shrinks every estimate by a quarter, and that shrinks the noise too.
Leave-one-out removes that bias, but with 4 samples and a 10% pass rate its baseline is itself noisy.

### Advantages and GAE (script 04)

| Measurement | Result |
|---|---|
| Mean \|A(s_t, a_t)\| by position, key token at t = 3 | 0.004, 0.007, 0.016, **0.272**, 0.000 |
| Mean \|A(s_t, a_t)\| by position, digit sum | 0.008, 0.021, 0.033, 0.055, 0.160 |
| Largest \|E_a[A(s, a)]\| over all states | 1.3e-17 |
| max \|GAE(gamma = 1, lambda = 1) - (R - V(s_t))\| | 6.0e-08 |

Relative gradient MSE, `||g - grad J||^2 / ||grad J||^2`, with GAE advantages and a critic equal to the exact V plus
noise of size sigma. "Redrawn" noise is new on every visit. "Persistent" noise is a fixed error per state.

| Critic | sigma | Rollouts N | lambda 0 | 0.5 | 0.8 | 0.95 | 1 |
|---|---|---|---|---|---|---|---|
| exact | 0 | 64 | **2.17** | 2.65 | 4.55 | 7.27 | 8.74 |
| redrawn | 0.08 | 64 | **3.45** | 3.72 | 5.54 | 8.09 | 9.45 |
| persistent | 0.08 | 64 | 4.42 | **3.76** | 5.28 | 7.92 | 9.38 |
| persistent | 0.08 | 1024 | 1.28 | 0.55 | **0.41** | 0.54 | 0.63 |
| redrawn | 0.2 | 64 | 10.60 | **8.86** | 10.13 | 12.61 | 14.02 |
| persistent | 0.2 | 64 | 16.34 | 9.64 | **9.21** | 11.44 | 12.88 |
| persistent | 0.2 | 1024 | 7.17 | 2.45 | 0.93 | **0.78** | 0.86 |

With a perfect critic, lambda = 0 wins: the TD error is then the exact advantage. With critic error, an interior
lambda wins. Persistent error is bias, so more rollouts do not wash it out. At lambda = 0 and sigma = 0.2, going from
64 to 1024 rollouts cuts the error only 2.3 times. Pure variance would fall 16 times. More rollouts also move the
best lambda to the right, toward the post's default of 0.95.

### Steps, collapse and the clip (scripts 05 and 06)

| Measurement | Result |
|---|---|
| Exact KL for 200 random parameter steps of the same length, max / min: untrained / after 100 RLOO steps | 3.6x / 12.5x |
| KL of one logit +1 on a two-way choice at p = 0.5 vs p = 0.99 | 0.111 vs 0.0026 |

RLOO with G = 8 for 300 steps. No KL penalty and no clip. Everything is exact. "Rare" is the mean of targets 4 and
20, which start at pass@16 = 0.20.

| Adam lr | pass@1 | pass@16 | Completion entropy (nats) | Largest KL per step | Rare pass@16 at the end |
|---|---|---|---|---|---|
| 1e-3 | 0.733 | 0.800 | 2.36 | 0.065 | 0.50 |
| 3e-3 | 0.599 | 0.600 | 1.24 | 0.074 | 0.00 |
| 1e-2 | 0.599 | 0.600 | 0.65 | 0.174 | 0.00 |
| 3e-2 | 0.600 | 0.600 | 0.05 | 1.979 | 0.00 |

The start is pass@1 0.049 and entropy 8.90 nats (uniform is 8.96). At every learning rate pass@1 goes up. At the
three larger rates the rare prompts go from pass@16 = 0.20 to exactly 0. The shared network learned "sum to the
middle" and stopped producing the rare answers at all. That is the post's "pass@1 improves while pass@k gets worse",
here on the hard prompts.

One batch of 80 rollouts reused for 16 epochs (Adam lr 1e-2, eps 0.2):

| Epoch | Clipped: tokens outside [0.8, 1.2] | Clipped: largest ratio | Clipped: exact KL | Unclipped: largest ratio | Unclipped: exact KL |
|---|---|---|---|---|---|
| 1 | 0.37 | 2.17 | 0.19 | 2.17 | 0.19 |
| 4 | 0.59 | 3.00 | 0.53 | 6.26 | 2.50 |
| 16 | 0.66 | 3.22 | 0.54 | 7.95 | 6.01 |

The clip stops the KL near 0.5 nats while the unclipped surrogate reaches 6. It does not keep the ratios inside the
band. The first Adam step alone pushed a ratio to 2.17, before the clip could act. Later, ratios near 3 remain,
because shared parameters keep moving tokens whose own gradient is already zero. In ordinary PPO
training (lr 3e-3, 4 epochs per batch) the clip fired on 1.7% of tokens on average, at most 16.5%.

### Estimators compared (script 07)

5 prompts x 8 rollouts per step, 200 steps, Adam lr 3e-3, seeds 0 to 2. pass@1 = 0.6 means the three common targets
are solved and neither rare one is.

| Method | Final pass@1, mean (min to max) | Steps to pass@1 >= 0.5 | Completion entropy |
|---|---|---|---|
| REINFORCE, no baseline | 0.726 (0.594 to 0.793) | 83 | 0.91 |
| batch-mean baseline | 0.661 (0.592 to 0.795) | 87 | 0.86 |
| RLOO (group leave-one-out) | 0.662 (0.596 to 0.794) | 97 | 1.26 |
| GRPO (group mean and std) | 0.657 (0.588 to 0.793) | 100 | 1.38 |
| learned critic + GAE (lambda 0.95) | 0.597 (0.596 to 0.597) | 97 | 0.81 |
| PPO, RLOO advantages, 4 epochs | 0.731 (0.599 to 0.798) | 30 | 0.37 |

Three seeds cannot separate the four critic-free estimators: their spread comes from whether a rare target gets
solved. The learned critic never solved a rare target in three seeds, and it costs a second network of the same
size. PPO reaches 0.5 three times sooner because it takes 4 gradient steps per batch. That is a per-rollout
comparison, not a per-compute one.

### Tensors and memory (script 08)

| Measurement | Result |
|---|---|
| Forgetting the post-EOS mask (62 of 192 positions): cosine of the gradients / relative difference | 0.905 / 0.48 |
| Weight of one token in a 100- vs 2,000-token answer: sequence mean / token mean | 20x / 1x |
| Logits for 256 sequences x 4,096 tokens x 128,000 vocabulary in bf16 | 268.4 GB |
| Chunked log-prob gather vs full, [8, 512, 8000] | max difference 0.0 |
| Weights + optimiser state at 7B: PPO with reward model / verifier / critic-free / no reference | 252 / 238 / 126 / 112 GB |

The memory rows are arithmetic with the post's 16 bytes per trainable parameter and 2 per frozen one. Nothing was
allocated at 7B.

### Cold start (script 09)

| Measurement | Result |
|---|---|
| Random init: exact p(target), 200 steps x 64 rollouts | p = 9.8e-11. 0 successes, 0 nonzero gradients, largest weight change 0.0 |
| After 20 SFT steps on noisy demonstrations: exact p(target) | 0.022 |
| The same RL loop afterwards, p(target) at step 0 / 10 / 20 / 100 | 0.022 / 0.544 / 0.888 / 0.994 |
| P(group has a success), measured vs 1 - (1 - p)^G, p = 0.01 and G = 4 / 16 / 64 | 0.042 vs 0.039 / 0.152 vs 0.149 / 0.472 vs 0.474 |

After the warm start, 41 of 100 RL steps had a zero gradient even though rollouts succeeded. At p = 0.99 a group
of 64 is all-correct about half the time, and then every RLOO weight is zero. That is the other degenerate case
from section 5. Which steps were zero was not logged.

### Sampler vs trainer numerics (script 10)

SmolLM2-135M, 16 completions x 256 tokens sampled in float32. Each completion is scored twice with the same weights:
token by token through the KV cache (sampler path), and in one teacher-forced forward (trainer path). Same CPU.

| Comparison | Per-token std of log-ratio | Largest per-token \|log-ratio\| | Std of summed log-ratio at T = 256 | sigma x sqrt(256) | Sequence ratio range at T = 256 |
|---|---|---|---|---|---|
| sampler vs trainer, float32 | 3.8e-06 | 2.3e-05 | 7.0e-05 | 6.1e-05 | 1.000 to 1.000 |
| sampler vs trainer, bfloat16 | 0.058 | 0.386 | 0.816 | 0.933 | 0.156 to 3.85 |
| sampler vs trainer, float16 | 0.0073 | 0.053 | 0.107 | 0.116 | 0.862 to 1.30 |
| trainer bfloat16 vs float32 | 0.047 | 0.311 | 0.689 | 0.745 | 0.175 to 2.83 |
| trainer float16 vs float32 | 0.0064 | 0.050 | 0.092 | 0.102 | 0.747 to 1.16 |

In bfloat16 the two paths disagree by 6% per token. Over 256 tokens the sequence-level ratio of the same policy
ranges from 0.16 to 3.9. The summed log-ratio spreads close to sigma x sqrt(T). Float16 cuts the per-token gap 8
times, which matches the direction of the FP16 result the post cites. These are plain PyTorch paths on one CPU.

## What was not measured

- Nothing here runs at the post's scale. The papers in section 14 compared vLLM or SGLang against FSDP or Megatron
  on large models. Script 10 shows the mechanism on a 135M model with plain PyTorch on both sides.
- The claim that generation dominates step time (section 11) was not timed. A GRU toy on a CPU would not say
  anything about GPU decoding.
- The memory table is arithmetic, not a measured allocation.
- The bits-per-step comparison in section 13 is the post's heuristic. It is not computed here.
- RL in pretraining (Reinforcement Pre-Training and related work) and asymmetric clip bounds are described in the
  post but not built here.
- The estimator comparison uses three seeds on one toy task. It shows the mechanisms. It does not rank the methods.
