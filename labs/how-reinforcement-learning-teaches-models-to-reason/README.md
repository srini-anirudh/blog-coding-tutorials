# Lab: how reinforcement learning teaches models to reason

This lab is the code companion to the theory post
[How Reinforcement Learning Teaches Models to Reason](https://srini-anirudh.github.io/blogs/how-reinforcement-learning-teaches-models-to-reason/).
It builds the post's central mechanisms from scratch and runs each one. The models are small enough that
every number comes out in seconds on a CPU.

Nothing here trains a language model. Each experiment is a toy, and each toy keeps only the quantity the post
argues about. Policies are softmax distributions over whole responses, or over a single "how long to write"
decision. The full GRPO loop on a real pretrained model is in
[C6, post-training](../../c06-post-training/) (`posttrain/grpo.py`, `scripts/train_rl.py`). This lab does not
repeat it.

## What it shows

1. A rule verifier and a surface-feature proxy score the same rollouts differently. The proxy likes a fluent wrong
   answer best.
2. Best-of-N against an imperfect scorer rises, peaks and then falls. Against an oracle it only rises.
3. Outcome rewards pay for lucky trajectories. Monte Carlo step labels measure the completer rather than the step.
4. PPO-based RLHF keeps 252 GB of persistent state for 7B models. The critic is 44% of it.
5. Any baseline that ignores the sample leaves the gradient unchanged. The group mean shrinks the gradient by a
   factor (G-1)/G. RLOO is exact.
6. With 0/1 rewards the GRPO advantages have closed forms. Groups where every sample agrees carry no gradient.
7. GRPO's 1/|o_i| normalization makes wrong answers longer, even when length cannot change correctness.
8. RL raises pass@1 and lowers pass@k at large k. The base model overtakes it.
9. SFT is a policy gradient with implicit reward 1/π. RL cannot reach a mode it never samples. SFT can.

## Layout

```
labs/how-reinforcement-learning-teaches-models-to-reason/
├── rlvrlab/
│   ├── verifier.py     # math_reward, format_reward, a surface-feature proxy, best-of-N simulation
│   ├── prm.py          # lucky trajectories, Monte Carlo step labels
│   ├── memory.py       # persistent-state accounting, bytes per trained parameter measured
│   ├── advantages.py   # REINFORCE, mean (Dr. GRPO), RLOO, GRPO advantages; binary closed forms; dynamic sampling
│   ├── bandit.py       # exact policy gradient on a bandit, Monte Carlo bias and variance of each estimator
│   ├── length_bias.py  # the length toy; grpo / dapo / dapo_batch / drgrpo loss weights
│   ├── sharpening.py   # strategy-mixture policy, exact pass@k, unbiased pass@k estimator
│   ├── operators.py    # SFT as a policy gradient, DFT weights, SFT vs RL on an out-of-support mode
│   └── common.py       # hardware line and result files
├── scripts/            # 01 to 09, one per mechanism
├── tests/              # 28 offline tests
└── results/            # what the scripts printed, as .txt and .json
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Only `torch` and `numpy` are used. There are no
downloads and no GPU code.

## Run everything

```bash
L=labs/how-reinforcement-learning-teaches-models-to-reason
python -m pytest $L -m "not slow and not gpu and not network" -o addopts="--import-mode=importlib" -q
python $L/scripts/01_reward_channels.py
python $L/scripts/02_best_of_n.py
python $L/scripts/03_process_labels.py
python $L/scripts/04_ppo_memory.py
python $L/scripts/05_baselines.py
python $L/scripts/06_binary_advantages.py
python $L/scripts/07_length_bias.py         # --steps 100 --seeds 1 for a quick run
python $L/scripts/08_pass_at_k.py
python $L/scripts/09_sft_vs_rl.py
```

Each script takes under 15 seconds on the CPU below. Each one writes `results/<script>.txt` and `.json`. Both files
start with the hardware line and the command.

## Post sections and scripts

| Post section | Script | Module |
|---|---|---|
| Two kinds of reward | `01_reward_channels.py` | `verifier.py` |
| The verifier era (best-of-N turnover) | `02_best_of_n.py` | `verifier.py` |
| Outcome and process | `03_process_labels.py` | `prm.py` |
| What PPO costs you | `04_ppo_memory.py` | `memory.py` |
| GRPO from first principles (baselines, the O(1/G) wrinkle) | `05_baselines.py` | `bandit.py`, `advantages.py` |
| What the algebra tells you about binary rewards | `06_binary_advantages.py` | `advantages.py` |
| Aha moment or artifact (the length bias) | `07_length_bias.py` | `length_bias.py` |
| Elicit or create (pass@k crossover) | `08_pass_at_k.py` | `sharpening.py` |
| SFT and RL as operators | `09_sft_vs_rl.py` | `operators.py` |

## Results

Every number below is from AMD EPYC 7R13, CPU only, torch 2.15.0, numpy 2.5.4. Scripts use 4 threads and tests
use 2. The machine was shared with other jobs. Only the verifier timing depends on load.

Tests: 28 passed in 5.3 s (`results/tests.txt`). There are no `slow`, `gpu` or `network` tests.

| Mechanism | Result |
|---|---|
| Four rollouts: math reward / surface proxy, fluent wrong answer | 0 / 0.84, the proxy's highest score |
| Four rollouts: math reward / surface proxy, terse right answer | 1 / 0.013 |
| Rule verifier cost | 1.55 µs per call, one thread |
| Rule verifier fooled by two boxed answers, last one right | reward 1 |
| Best-of-N, heavy-tailed scorer: N=1 / peak / N=2048 | 0.262 / 0.824 at N=32 / 0.403 |
| Best-of-N, oracle (pass@N) at N=2048 | 1.0 |
| Rewarded trajectories with a wrong step (15% of broken ones land right) | 11.9% |
| MC hard labels, strong completer: bad steps labelled bad | 1.7% |
| MC hard labels, weak completer: bad steps labelled bad | 85.1% |
| MC soft labels, weak completer: good steps labelled good | 36.3% |
| Bytes per trained parameter after one AdamW step, fp32 / bf16 with fp32 master | 16.0 / 16.0 (measured) |
| Persistent state, 7B, PPO-based RLHF / GRPO with a verifier | 252 GB / 126 GB (arithmetic) |
| Gradient scale of the group-mean baseline, G = 2 / 8 / 64 | 0.503 / 0.876 / 0.984 (predicted 0.5 / 0.875 / 0.984) |
| Gradient scale of RLOO, G = 8 | 1.001 |
| Variance, G = 8: REINFORCE / RLOO / mean | 2.06 / 1.38 / 1.06 (units of the squared gradient norm) |
| GRPO advantages at p = 7/8, population std | +0.378 right, −2.646 wrong |
| Same, with the n−1 std that TRL and C6 use | +0.354 right, −2.475 wrong |
| Chance a group of 8 has any gradient, p = 0.05 | 0.337 |
| Groups with a gradient, solve rate ~ Beta(0.4, 0.4), G = 8 | 53.2% (15.0 rollouts per useful group) |
| Weight of a 1/8 prompt against a 4/8 prompt, GRPO / mean only | 0.661 / 0.438 |
| Length bias at init, push on the "guesses" stop logit (z-score), grpo / dapo / dapo_batch / drgrpo | −262 / −87 / −0.9 / +0.4 |
| Mean length of wrong answers, step 0 → 400, 3 seeds, grpo | 41 → 622, 664, 668 |
| Same, dapo (normalized per group) | 41 → 704, 745, 744 |
| Same, dapo_batch / drgrpo | 33 to 69 / 29 to 56 |
| Accuracy in the length toy, all runs | 0.49, unchanged |
| pass@1, base → RL (β = 0) | 0.115 → 0.466 |
| pass@64, base → RL (β = 0) | 0.931 → 0.589 |
| k at which the base model overtakes RL, β = 0 / 0.1 / 1.0 | 16 / 16 / 32 |
| pass@256 over training, β = 0, steps 0 / 50 / 300 | 0.950 / 0.801 / 0.601 |
| Unbiased pass@k estimator, n = 256, against exact pass@64 | 0.932 against 0.931 |
| SFT as an on-policy gradient, demonstration at π = 2.7e-7: error with 1e5 / 1e7 samples | 100% / 27% |
| 300 steps, π(target): SFT / RL | 0.997 / 2.2e-9 (RL sampled it 0 times in 2400) |

### How to read the surprising rows

The length toy's state "knows" is right 90% of the time and "guesses" is right 10%. The policy sees which state
it is in. Length never changes correctness. So the growth of wrong answers under `grpo` comes from the estimator
alone. The bias table measures this with no optimizer in the loop.

The `dapo` row normalizes by the token count of one group, as the DAPO paper's formula is written. In this toy it
still has about a third of GRPO's bias, measured in units of noise. A long response enlarges its own denominator,
so its push toward stopping is diluted. Normalizing over the whole batch (`dapo_batch`, as most trainers do) or by
a constant (`drgrpo`) removes the bias. This is a finding about the toy. It is not a claim about DAPO's runs.

The `dapo_batch` and `drgrpo` lengths still move between 29 and 69. That is noise. Adam rescales every gradient
to a similar step size, so a zero-mean gradient still causes a random walk. Adam also turns any small consistent
bias into a full-speed drift. Read the bias table for the size of the effect, and the training table for its
direction.

In `05_baselines.py`, plain GRPO has a gradient scale near 1.6 to 2.0, not 1. With one prompt, dividing by the
group std rescales the step without turning it. The cosine stays 1. Across many prompts the std changes the weight
of each prompt, which is the difficulty weighting in `06_binary_advantages.py`.

In `09_sft_vs_rl.py`, SFT lowers some modes that are not the demonstration. It lowers the most likely ones,
whether they are right or wrong, because the softmax must stay normalized. RL lowers exactly the wrong ones.

## What was not measured

- No language model is trained here. The post's large-scale results are not reproduced: DeepSeek-R1 and R1-Zero,
  the Spurious Rewards table, Lightman et al.'s 78% on MATH, Yue et al.'s pass@k crossover on real models, ProRL,
  and Cobbe et al.'s 30× verifier result. The toys show the mechanism each result is about.
- The best-of-N scorer, the completer strengths and the length toy's accuracies are chosen parameters. They are
  not fitted to any paper.
- The 252 GB and 126 GB figures are arithmetic on 16 and 2 bytes per parameter. Only the 16 bytes per trained
  parameter is measured, on a small module. Activations and KV cache are left out.
- The surface proxy is written by hand. It is not a trained reward model.
- No GPU numbers. Nothing in this lab needs a GPU.
