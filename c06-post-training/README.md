# C6: Post-Train an LLM

One base model, one codebase, three post-training stages: SFT, then DPO, then GRPO. The package `posttrain` writes
each stage by hand in plain PyTorch, then checks it against the library that does the same job (the Hugging Face chat
template, PEFT, TRL's `DPOTrainer` and TRL's GRPO loss).

Tutorial: https://srini-anirudh.github.io/blogs/post-train-an-llm/

The base model is [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), the pretrained base, not the
Instruct model. A tiny model trained from scratch in C5 is not capable enough for preference learning or RL. The
`--init random` run below shows how far it falls short.

One task carries the whole tutorial: synthetic arithmetic conversations (`posttrain/tasks.py`). We generate the data,
so every label and every reward is exact. SFT teaches the answer format, DPO prefers correct answers over near-miss
wrong ones, and GRPO optimises the reward `1 if "The answer is N." has the right N else 0`.

## Layout

```
c06-post-training/
├── posttrain/
│   ├── tasks.py          # synthetic arithmetic problems, exact reward, near-miss wrong answers
│   ├── chat.py           # ChatML template, assistant-only labels, token-by-token mask table
│   ├── packing.py        # padding vs packing, block-causal mask, position ids that restart
│   ├── sft.py            # masked LM loss, warmup + cosine schedule, the SFT loop
│   ├── lora.py           # LoRALinear, apply_lora, merge_lora (checked against PEFT)
│   ├── logprobs.py       # per-token and per-sequence log pi(y|x)
│   ├── dpo.py            # preference pairs (synthetic and on-policy), batch layout, DPO loss
│   ├── reward_model.py   # scalar head on the last real token, Bradley-Terry loss
│   ├── generation.py     # left-padded sampling, completion masks, scoring completions
│   ├── reinforce.py      # REINFORCE loss, four baselines, gradient-noise statistics
│   ├── grpo.py           # group advantages, clipped surrogate, k3 KL, grpo/dapo averaging
│   ├── rollout.py        # generation workers, policy versions, reward, training batch
│   ├── evaluate.py       # greedy accuracy on held-out prompts, per problem kind
│   ├── models.py         # load the policy, a frozen reference, random init, a tiny test model
│   └── utils.py          # device, threads, memory, hardware label, results
├── scripts/
│   ├── inspect_mask.py        # one conversation, token by token, with its labels
│   ├── train_sft.py           # Stage 1: full FT, LoRA, packed, random init
│   ├── train_reward_model.py  # Stage 3: pairwise reward model
│   ├── train_dpo.py           # Stage 2: DPO with synthetic or on-policy pairs
│   ├── reinforce_variance.py  # Stage 4: how noisy each policy-gradient estimator is
│   ├── train_rl.py            # Stages 4-7: REINFORCE or GRPO through the rollout engine
│   ├── inspect_state.py       # the tensors behind one DPO pair and one GRPO group
│   └── break_it.py            # seven post-training bugs, run on purpose
├── benchmarks/
│   ├── bench_packing.py   # padded vs packed batches: real-token fraction, target tokens/sec
│   ├── bench_memory.py    # full FT vs LoRA: peak memory and sec/step, one process per mode
│   └── bench_rollout.py   # rollouts/sec: sequential, batched, two workers; weight-sync time
├── results/              # CPU outputs (AMD EPYC 7R13, 8 threads)
│   └── h100/             # GPU outputs (NVIDIA H100 80GB HBM3), same file names
└── tests/                # 48 tests: 38 offline, 10 that download SmolLM2-135M
```

## Run

From the repository root, after `pip install -e ".[dev]"`. Everything runs on CPU, CUDA or Apple MPS. The device is
picked as cuda > mps > cpu, or set with `POSTTRAIN_DEVICE`. Scripts use `$OMP_NUM_THREADS` CPU threads, or 8 if it
is unset. Run the scripts from `c06-post-training/`.

```bash
# Tests (from the repository root).
python -m pytest c06-post-training -m "not slow and not gpu and not network"   # 38 tests, about 25 s
python -m pytest c06-post-training -m network                                  # 10 tests: HF template, TRL, rollouts

cd c06-post-training

# Stage 1: SFT. Each run writes results/<tag>.json and a checkpoint under checkpoints/<tag>.
python scripts/inspect_mask.py
python scripts/train_sft.py --mode full --steps 500                        # -> checkpoints/sft-full, used by later stages
python scripts/train_sft.py --mode lora --steps 500
python scripts/train_sft.py --mode full --pack --batch-size 5 --steps 500
python scripts/train_sft.py --mode full --init random --steps 500

# Stage 3 (optional) and Stage 2. Both start from checkpoints/sft-full.
python scripts/train_reward_model.py --steps 300
python scripts/train_reward_model.py --steps 1500 --batch-size 16 --tag reward_model_long
python scripts/train_dpo.py --pairs synthetic --steps 200
python scripts/train_dpo.py --pairs on-policy --steps 200
python scripts/train_dpo.py --pairs synthetic --steps 200 --lr 1e-6 --tag dpo-synthetic-lr1e-6
python scripts/train_dpo.py --pairs on-policy --steps 200 --lr 1e-6 --tag dpo-on-policy-lr1e-6

# Stages 4-7.
python scripts/reinforce_variance.py --trials 12 --samples 32
python scripts/train_rl.py --algo grpo --steps 100 --eval-every 20 --seed 0 --tag grpo
python scripts/train_rl.py --algo reinforce --steps 100 --eval-every 20 --seed 0 --tag reinforce
python scripts/train_rl.py --algo grpo --steps 100 --eval-every 20 --seed 0 --lr 1e-6 --tag grpo-lr1e-6
python scripts/train_rl.py --algo reinforce --steps 100 --eval-every 20 --seed 0 --lr 1e-6 --tag reinforce-lr1e-6
# Seeds 1 and 2: the same two commands at lr 1e-5 with --seed 1 --tag grpo-seed1 (and reinforce-seed1, -seed2).

# Look inside, and break it.
python scripts/inspect_state.py
python scripts/break_it.py

# Benchmarks. They print tables.
python benchmarks/bench_packing.py --steps 12
python benchmarks/bench_memory.py --steps 10                                # CPU: add --batch-size 2 --out bench_memory_b2.json for the small batch
#                                                                         # H100: --steps 20
python benchmarks/bench_rollout.py
```

Set `POSTTRAIN_RESULTS=h100` to write results to `results/h100/` instead of `results/`. The GPU runs used it.

## Results

Every number below is a real run. Accuracy is greedy decoding on 200 held-out prompts (50 each of add, sub, chain
and mul). Each script draws its own held-out prompts with its own seed, and no training run sees them, so compare a
stage only with its own "before" number. The base model scores 0% on the SFT set because it never writes
"The answer is N.".

The SFT, LoRA, DPO and reward-model runs used one seed. On 200 prompts, one standard error is about 3 points, so read gaps
under about 6 points as noise.

### Stage 1: SFT

| Run | Hardware | Steps | Trainable params | Accuracy | add / sub / chain / mul | Train time | Checkpoint |
|---|---|---|---|---|---|---|---|
| Base model, no SFT | both | 0 | - | 0.0% | 0 / 0 / 0 / 0 | - | - |
| Full FT, padded | AMD EPYC 7R13, 8 threads | 500 | 134.5M | 74.5% | 98 / 86 / 68 / 46 | 2464 s | 513 MiB |
| Full FT, padded | NVIDIA H100 | 500 | 134.5M | 74.5% | 96 / 86 / 68 / 48 | 133 s | 513 MiB |
| LoRA r=16, all linear layers | NVIDIA H100 | 500 | 4.9M (3.6%) | 75.0% | 96 / 84 / 72 / 48 | 208 s | 18.7 MiB |
| Full FT, packed, 5 rows of 256 | NVIDIA H100 | 500 | 134.5M | 69.0% | 96 / 80 / 48 / 52 | 98 s | 513 MiB |
| Full FT from random weights | NVIDIA H100 | 500 | 134.5M | 1.5% | 2 / 4 / 0 / 0 | 112 s | 513 MiB |

- CPU and GPU full FT reach the same accuracy with nearly the same learning curve (`history` in the two
  `sft-full.json`). The H100 was about 19 times faster (133 s against 2464 s). Train time includes the evals every 50 steps. An earlier identical H100 run took 102 s, so the GPU timing varies by about 30% between runs; the accuracy was identical.
- The packed run took the same number of steps but saw 10% fewer target tokens (206k vs 230k), so its lower
  score does not show that packing hurts quality. Its speed is measured separately, below.
- From random weights the model learns the answer format (100% formatted) but not the arithmetic. Its loss stalls
  at about 0.35, against about 0.02 for the pretrained base. The capability comes from pretraining. SFT only
  surfaces it.

### Full FT vs LoRA: memory and speed (`bench_memory.py`)

Each mode runs in a fresh process that only loads the model and takes optimizer steps, so no eval inflates the peak.

| Hardware | Batch | Memory measured | Full FT peak | LoRA peak | Full FT sec/step | LoRA sec/step |
|---|---|---|---|---|---|---|
| NVIDIA H100 | 16 | peak CUDA memory allocated | 5938 MiB | 4695 MiB | 0.24 | 0.31 |
| AMD EPYC 7R13, 8 threads | 16 | peak resident memory of the process | 7045 MiB | 7381 MiB | 4.70 | 8.25 |
| AMD EPYC 7R13, 8 threads | 2 | peak resident memory of the process | 3165 MiB | 1897 MiB | 1.09 | 0.73 |

- LoRA does not store gradients or AdamW state for the 130M frozen weights. That saving is fixed. Activations grow
  with the batch and LoRA does not shrink them. On the H100 at batch 16, LoRA's peak was 21% lower.
- On CPU the order flips with the batch: LoRA's peak was 5% higher at batch 16 and 40% lower at batch 2. Its first
  forward pass already reached 7001 MiB against 5432 MiB for full FT (`forward_mb` in `bench_memory.json`). On
  the H100 the same forward used less memory with LoRA (4275 vs 4516 MiB). The CPU number is the resident memory of
  the whole process, including whatever the allocator keeps, so these runs do not pin down the cause.
- Seconds per step are noisy on both machines, so do not read any column as a speed ratio. The same H100
  benchmark gave full FT 0.12 s in one run and 0.24 s in the next, with identical memory. The CPU benchmark gave
  LoRA 5.51 s and then 8.25 s. The table shows the later run in each case, because that is the saved file.

### Padding vs packing (`bench_packing.py`)

2000 conversations of 62 to 124 tokens. Padding in batches of 8 makes 63.7% of the slots real tokens. Packing into
rows of 256 makes 89.8% real. Forward and backward only:

| Hardware | Batching | sec/step | Target tokens/step | Target tokens/sec |
|---|---|---|---|---|
| AMD EPYC 7R13, 8 threads | padded, 8 rows | 2.25 | 222 | 99 |
| AMD EPYC 7R13, 8 threads | packed, 4 rows of 256 | 2.48 | 331 | 134 |
| NVIDIA H100 | padded, 8 rows | 0.17 | 224 | 1332 |
| NVIDIA H100 | packed, 4 rows of 256 | 0.16 | 329 | 2118 |

Packing is only correct with a block-causal mask and position ids that restart at 0 for each conversation. Without
the mask, the logits of the second conversation change by up to 39.75 (`break_it.py`, below).

### Stage 3: reward model, and Stage 2: DPO

Both stages start from the SFT checkpoint trained on the same hardware (`checkpoints/sft-full`). All numbers here
are NVIDIA H100, one seed. Each script draws its own 200 held-out prompts, so the "before" accuracy differs from
the SFT table (79.5% here, 74.5% there) while the checkpoint is the same. Compare a stage only with its own "before".

**Reward model** (`train_reward_model.py`). A scalar head on the last real token, trained with the Bradley-Terry loss on
near-miss pairs, scored on 200 held-out pairs:

| Steps | Batch | Held-out pair accuracy | Last training accuracy |
|---|---|---|---|
| 300 | 4 | 55.5% | noisy, 0.25 to 1.0 |
| 1500 | 16 | 59.5% | 1.0 |

This is a weak result. Chance is 50%. With ten times more training the model memorises the training pairs and still
cannot tell a right sum from a wrong one on new problems. To score an answer it has to do the arithmetic itself,
from a single hidden state. A reward model is the wrong tool for a task with an exact checker, which is why Stage 6
uses the exact reward. The "before" figure of 0.000 is a tie: the head starts at zero, so `chosen > rejected` is
never true.

**DPO** (`train_dpo.py`). 200 steps, batch 4, beta 0.1, full fine-tuning, greedy accuracy on 200 held-out prompts:

| Pairs | Learning rate | Pairs used | Accuracy before | Accuracy after | add / sub / chain / mul after | Final pair margin |
|---|---|---|---|---|---|---|
| synthetic near-miss | 5e-6 | 800 | 79.5% | 44.5% | 88 / 46 / 14 / 30 | 0.62 |
| on-policy | 5e-6 | 305 of 800 prompts | 79.5% | 42.0% | 70 / 36 / 30 / 32 | 0.74 |
| synthetic near-miss | 1e-6 | 800 | 79.5% | 66.5% | 96 / 70 / 52 / 48 | 0.30 |
| on-policy | 1e-6 | 305 | 79.5% | 65.5% | 96 / 66 / 48 / 52 | 0.19 |

DPO made the model worse in all four runs, and the pairs were always ranked correctly during training (pair accuracy
0.75 to 1.0, margin above 0). The loss matches TRL's `DPOTrainer` on the same batch (`tests/test_dpo.py`), so this is
not a loss bug. A likely cause, which I did not measure, is the known likelihood-displacement effect: the chosen and
rejected answers share almost every token, so pushing down the rejected one can pull down the chosen one too, and
the margin still grows. `inspect_state.py` shows how little separates a pair: only the wrong digits differ, and the
rest of the tokens have a log-prob near 0. A lower learning rate hurts less and still hurts. I did not tune further. Things worth
trying are a smaller beta, an added SFT term on the chosen answer, or fewer steps. Only 305 of 800 prompts gave an
on-policy pair, because the SFT model answered the rest all right or all wrong in four samples.

### Stages 4-7: REINFORCE, GRPO and the rollout engine

Both algorithms start from the SFT checkpoint (80.0% greedy accuracy on this script's 200 held-out prompts), spend 32
rollouts per step for 100 steps, and use AdamW with no weight decay. GRPO uses 8 prompts x 4 samples, 2 inner steps per
batch, clip 0.2 and KL coefficient 0.04. REINFORCE uses 32 prompts x 1 sample and no baseline. All on one NVIDIA H100.

| Algorithm | Learning rate | Seeds | Greedy accuracy after 100 steps (start 80.0%) | Mean KL to reference, last 10 steps |
|---|---|---|---|---|
| GRPO | 1e-5 | 3 | 64.5%, 65.5%, 59.0% | 0.034, 0.037, 0.042 |
| REINFORCE | 1e-5 | 3 | 73.0%, 76.0%, 79.0% | not computed |
| GRPO | 1e-6 | 1 | 78.5% | 0.006 |
| REINFORCE | 1e-6 | 1 | 81.0% | not computed |

This is a weak result and I report it as one. Neither algorithm improves the model in 100 steps.
- At lr 1e-5, GRPO is worse than where it started in all three seeds, by 14.5 to 21 points. REINFORCE ends 7, 4 and 1
  points below the start. One run's standard error is about 3 points.
- At lr 1e-6 both stay within noise of the start (78.5% and 81.0%). GRPO's mean KL to the reference is 0.006 there,
  against 0.034 to 0.042 at lr 1e-5, so the policy barely moves, and the accuracy follows.
- The training reward printed by the script is sampled at temperature 1 on 32 rollouts, so it moves by about 0.1
  from step to step. It is not a good progress signal. Use the greedy evals in each `history`.
- I did not run a search over learning rates, batch sizes or group sizes. The SFT model already solves 80% of
  problems, so many groups have all-equal rewards and zero advantage. A harder task, a weaker start, a larger group or
  more steps are the obvious next things to try. The full-size runs in the tutorial's exercises are the place for that.
- A comparison of GRPO and REINFORCE from these runs is not safe: at the same learning rate GRPO did worse, but each
  algorithm has only three seeds and one setting.

Gradient noise of four policy-gradient estimators (`reinforce_variance.py`, CPU, SFT checkpoint trained on CPU). Each
estimate uses 32 rollouts, and each estimator gets 12 independent estimates of the gradient of the last block
(3.54M numbers):

| Estimator | Prompts x samples | Grad norm | Pairwise cosine | SNR |
|---|---|---|---|---|
| R (no baseline) | 32 x 1 | 0.73 | 0.037 | 0.147 |
| R - batch mean | 32 x 1 | 1.45 | -0.007 | 0.080 |
| R - group mean | 8 x 4 | 1.21 | 0.007 | 0.104 |
| (R - group mean) / group std | 8 x 4 | 2.29 | 0.005 | 0.101 |

The same measurement on the H100, with the checkpoint trained on the H100:

| Estimator | Prompts x samples | Grad norm | Pairwise cosine | SNR |
|---|---|---|---|---|
| R (no baseline) | 32 x 1 | 0.87 | 0.042 | 0.115 |
| R - batch mean | 32 x 1 | 1.37 | 0.014 | 0.108 |
| R - group mean | 8 x 4 | 1.02 | 0.002 | 0.106 |
| (R - group mean) / group std | 8 x 4 | 1.86 | 0.002 | 0.106 |

Pure noise gives an SNR of about 1/11 = 0.091 with 12 estimates, so every estimator on both machines sits close to
noise: two independent 32-rollout gradients barely agree on a direction. That is the point of the measurement. It
does not rank the estimators. The CPU run had the highest SNR for the version with no baseline, and the H100 run
did not, which suggests the CPU ordering was luck. A rerun of the CPU measurement gave the same numbers to every
digit, so the difference comes from the checkpoint or the hardware, not from the sampling.

Rollouts per second (`bench_rollout.py`, AMD EPYC 7R13 with 8 threads, 64 rollouts of up to 48 new tokens):

| Engine | sec | Rollouts/sec |
|---|---|---|
| Sequential, batch 4 | 36.04 | 1.8 |
| Batched, batch 64 | 9.38 | 6.8 |
| 2 threaded workers, batch 32 each | 8.87 | 7.2 |

Pushing new weights to two worker replicas took 191 ms. Scoring with the exact reward took under 1 ms.

The same benchmark on the H100:

| Engine | sec | Rollouts/sec |
|---|---|---|
| Sequential, batch 4 | 16.55 | 3.9 |
| Batched, batch 64 | 1.90 | 33.7 |
| 2 threaded workers, batch 32 each | 2.58 | 24.8 |

Batching is worth 8.6 times on the H100 and 3.8 times on CPU. Two workers helped a little on CPU (7.2 against 6.8
rollouts/s) and hurt on the H100 (24.8 against 33.7), because both workers share one GPU and one Python process.
Weight sync to two replicas took 14 ms. The two-worker engine only pays off when replicas sit on different devices
or the environment is slow, as in agent RL (C12).

### Break it (`break_it.py`, AMD EPYC 7R13, 8 threads)

| Bug | What it prints |
|---|---|
| Pad token = `<|im_end|>`, and the collator masks pad ids | stop-token targets in a conversation: 1 correct, 0 with the bug, so the model never learns to stop |
| Packing without a block mask | max logit difference from running alone: 0.00000 with the mask, 39.754 without |
| DPO reference = the policy itself | loss 0.6931 = log 2 and margin 0 forever. The grad norm (3.97) looks normal |
| DPO reference with gradients on | 3.95 of gradient norm lands on the reference, which must get 0 |
| Mean instead of sum of token log-probs | margin shrinks by the reply length (about 23 tokens): 0.0031 vs 0.0001. Grad norm 3.96 vs 0.19 |
| Off-by-one shift in the completion mask | mean log pi(y\|x): -4.23 right, -20.04 with the mask one token late |
| GRPO old log-probs recomputed each inner step | ratio stays 1.000 and clipping never fires. Fixed old log-probs: ratio 0.98, 3-4% clipped |

The H100 run (`results/h100/break_it.txt`) prints the same pattern with slightly different values: logit difference 0.00006 with the mask and 39.616 without, and the same loss 0.6931 for the self-reference bug.

## Tests

`results/pytest.txt` has the per-test output. 38 offline tests and 10 network tests pass.

- Chat and packing: manual token ids equal `apply_chat_template`. The manual mask equals Hugging Face's
  `{% generation %}` assistant mask. Packed logits equal unpacked logits. Naive packing leaks.
- LoRA: starts at the base model, trains only the adapters, matches PEFT with the same A and B, and merging keeps
  the outputs.
- DPO: sequence log-probs match an explicit loop and ignore prompt and padding. Equal policy and reference give
  log 2. The reference gets no gradient. The loss matches TRL's `DPOTrainer` on the same batch.
- GRPO: advantages by hand and by TRL's formula. Clipping, masks, k3 KL and grpo/dapo averaging. The loss matches TRL's
  GRPO loss on the same tensors. The rollout engine builds a consistent batch. Worker replicas change only on sync.

## Hardware and what is not covered

- CPU: AMD EPYC 7R13, 8 threads, on a shared machine, torch 2.15.0. GPU: one NVIDIA H100 80GB HBM3, torch 2.15.0,
  CUDA 13.2, fp32 weights. Each JSON file names its hardware in its `hardware` field.
- The CPU `break_it`, `inspect_state`, `bench_rollout` and `reinforce_variance` results used the SFT checkpoint
  trained on CPU. Every result in `results/h100/` used the checkpoint trained on the H100. The reward model, DPO,
  GRPO and REINFORCE results exist only for the H100, because they would take hours on this CPU.
- GPU timings come from a node shared with other jobs. The same H100 benchmark gave full FT 0.12 s per step once and
  0.24 s another time, and the same SFT run took 102 s once and 133 s another time.
- Not covered: multi-GPU training, mixed precision (all runs are fp32), long-context SFT, and any RL run with
  more than 100 steps or more than 32 rollouts per step.
- Packages: transformers 5.22.2, peft 0.21.2, trl 1.17.1, datasets 5.1.0. No package beyond the repository's
  `pyproject.toml` is needed.
