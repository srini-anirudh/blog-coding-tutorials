# Lab: Thinking in Tokens

Runnable code for the post [Thinking in Tokens: What Reasoning Models Actually Do](https://srini-anirudh.github.io/blogs/thinking-in-tokens/).

The post argues that a reasoning model is the same Transformer as before. What changed is what it writes before it
answers. This lab makes that claim concrete in five pieces:

1. **A written trace buys depth.** A 2-layer Transformer is trained from scratch on parity. It answers either in one
   token or after writing the running parity. Same model, same data, different output format.
2. **The bill for thinking.** The post's FLOP and KV-cache formulas are checked against torch's FLOP counter on a
   real decoder. Every number the post quotes for a 32B-class model is recomputed from its config.
3. **When voting helps.** Self-consistency is simulated from stated answer distributions, to show it recovers the
   *mode* of the answer distribution, not the majority.
4. **Voting on a real model.** Qwen3-0.6B answers GSM8K questions 16 times each. Plurality voting is compared with
   a single sample and with a perfect verifier.
5. **Thinking controls are edits to the token stream.** The empty think block, the `/no_think` flag, Qwen3's
   interrupt-and-salvage budget and s1's "Wait" are each a few lines of decode-loop code, run on Qwen3-0.6B.

The lab reuses the decoder from tutorial C4 (`c04-llm/tinyllm`) instead of building a new one.

## Layout

```
labs/thinking-in-tokens/
├── thinktok/
│   ├── parity.py    # parity data in two formats, training and greedy evaluation on TinyLLM
│   ├── cost.py      # FLOP, KV-cache and decode-latency formulas; check against torch's FLOP counter
│   ├── vote.py      # plurality vote, vote@k, pass@k, simulated voting from a fixed answer distribution
│   ├── budget.py    # hard and soft thinking switches, Qwen3 budget, s1 "Wait" (model-agnostic)
│   ├── hf.py        # the Hugging Face `generate` adapter the budget controller calls
│   ├── gsm8k.py     # GSM8K loading and answer extraction
│   └── util.py      # hardware labels for result files
├── scripts/         # one script per experiment, listed below
├── results/         # real outputs; each file starts with its hardware and command
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Scripts 01 to 03 run on CPU. Script 04 downloads
Qwen3-0.6B (about 1.5 GB) and GSM8K from the Hugging Face Hub and needs a GPU to finish in minutes.

## Run everything

```bash
# Tests. The offline set uses a scripted fake model, so it needs no download.
python -m pytest labs/thinking-in-tokens -m "not slow and not gpu and not network"
python -m pytest labs/thinking-in-tokens -m network          # Qwen3 tokenizer from the Hub
python -m pytest labs/thinking-in-tokens -m gpu              # Qwen3-0.6B on CUDA

# CPU experiments.
python labs/thinking-in-tokens/scripts/01_parity_depth.py    # about 1 hour with 4 runs in parallel on a busy node; --quick takes about 1 min
python labs/thinking-in-tokens/scripts/02_cost_of_thought.py # about 1.5 min
python labs/thinking-in-tokens/scripts/03_vote_theory.py     # about 15 s

# GPU experiments (one H100 each).
python labs/thinking-in-tokens/scripts/04_qwen3_runs.py --part modes vote
python labs/thinking-in-tokens/scripts/04_qwen3_runs.py --part budget waits
```

## Results

Every number names its hardware. Raw outputs are in `results/`, each starting with a hardware line and the command.
Tests: 26 offline, 1 network and 1 GPU test pass (`results/tests.txt`).

### 1. Parity with and without a written trace (AMD EPYC 7R13, 2 threads per run, 4 runs in parallel)

A 2-layer TinyLLM (hidden size 128) is trained from scratch for 2,000 AdamW steps on fresh random strings. Accuracy is
measured on 2,000 held-out strings. Source: `results/01_parity_depth.md`.

| Bits n | Direct answer, 2 layers | With running-parity trace, 2 layers |
|---|---|---|
| 4 | 1.000 | 1.000 |
| 8 | 0.514 | 1.000 |
| 16 | 0.507 | 1.000 |
| 32 | 0.521 | 1.000 |
| 16, direct, 6 layers | 0.507 | not run |

The trace models reach 0.99 by the first evaluation, at step 500. Evaluation happens every 500 steps, so that is a
bound, not the exact step. The direct models sit at chance for the whole run, including the 6-layer model.

How to read it. The model, data and optimizer are identical. Only the output format differs. With the trace, each
token only XORs one new bit into the previous token, so the depth needed per token is constant. Without it, one
forward pass must combine all n bits.

What this does not show. It is one seed per cell and 2,000 steps. A direct model at chance after 2,000 steps shows
that gradient descent did not find parity here. It does not show that the architecture cannot represent it. The
circuit-complexity separation the post cites is a statement about representability.

### 2. The bill (AMD EPYC 7R13, 2 threads; arithmetic rows need no hardware)

`results/02_cost_of_thought.md`. On a real TinyLLM decode loop, torch's FLOP counter equals
2NK + 4Ld(PK + K(K+1)/2) exactly, for four (P, K) settings. The KV-cache bytes the cache holds equal the formula.
The post writes the attention pair count as K^2/2, which is within 0.1% of the exact K(K+1)/2 form in every setting tried.

The post's quoted numbers, recomputed from a 64-layer, 8-KV-head, head-size-128 config (Qwen2.5-32B-like):

| Quantity | Post | Recomputed |
|---|---|---|
| KV cache per token (bf16) | 256 KiB | 256 KiB |
| KV cache for 10,000 thinking tokens | about 2.6 GB | 2.62 GB |
| Decode floor per token, 64 GB of weights at 3.3 TB/s | about 19 ms | 19.4 ms |
| 10,000 thinking tokens at that floor | three minutes | 3.2 min |
| Attention / weight FLOPs per token at 1k context | a few per cent | 2.2% |
| Attention / weight FLOPs per token at 32k context | roughly comparable | 69% |
| Context where the two terms are equal | not stated | about 48k tokens |
| log2 of Qwen3's vocabulary (151,669) | 17.2 bits | 17.21 bits |

These are computed from formulas and a config, not measured on a 32B model. The decode floor ignores KV-cache reads,
so real batch-1 latency is higher.

### 3. When voting helps (AMD EPYC 7R13; simulation, not model output)

`results/03_vote_theory.md`. P(plurality of N draws is the right answer), 20,000 trials per cell.

| Answer distribution | Right is the mode? | N=1 | N=5 | N=21 | N=101 |
|---|---|---|---|---|---|
| Right 40%, the other 60% spread over 20 wrong answers | yes | 0.399 | 0.680 | 0.989 | 1.000 |
| Right 40%, one wrong answer at 45% | no | 0.399 | 0.441 | 0.402 | 0.297 |
| Right 60%, one wrong answer at 40% | yes | 0.600 | 0.684 | 0.827 | 0.979 |
| Same wrong answer 70%, right 20% | no | 0.203 | 0.097 | 0.005 | 0.000 |

A model that is right 40% of the time is fully recovered by voting when its errors are scattered. The same 40% is
made worse by voting when one wrong answer is more common. This is the post's "mode, not majority" condition.

### 4. Voting, modes and budgets on Qwen3-0.6B (NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2)

Qwen3-0.6B on GSM8K test questions drawn with a fixed seed. Prompt: question plus "reason step by step, put the
final answer in \boxed{}". Sampling settings are in each result file.

**Mode controls** (`results/04_modes.md`, 100 questions, does the model open with an empty think block?)

| Control | Thinking off |
|---|---|
| Default template | 0/100 |
| `/think` appended | 0/100 |
| `/no_think` appended (soft) | 100/100 |
| `enable_thinking=False` (hard prefill) | 100/100 (by construction) |

The soft flag obeyed on all 100 prompts for this model. The hard prefill cannot fail, because the closed block is
already in the prompt. The soft flag gives no such guarantee, and 100 prompts do not bound how often it fails.

**Self-consistency without thinking** (`results/04_vote.md`, 200 questions, 16 samples each, non-thinking mode)

| k samples | vote@k | pass@k (oracle verifier) |
|---|---|---|
| 1 | 0.618 | 0.619 |
| 4 | 0.665 | 0.793 |
| 8 | 0.688 | 0.856 |
| 16 | 0.715 | 0.915 |

On 140 of 200 questions the right answer is the mode of the 16 samples. On 15 of those it appears in at most half
the samples, so the vote recovers a minority. On 33 questions a wrong answer is the mode with at least half the
samples, so the vote confirms an error. The gap between vote@16 (0.715) and pass@16 (0.915) is the room a verifier
would add, which is the post's generator-verifier gap.

**Thinking budget with interrupt-and-salvage** (`results/04_budget.md`, 100 questions, thinking mode)

| Budget (tokens) | Accuracy | Mean thinking tokens | Cut off by the budget |
|---|---|---|---|
| 0 (hard switch) | 0.610 | 0 | 0% |
| 64 | 0.680 | 64 | 100% |
| 128 | 0.590 | 128 | 100% |
| 256 | 0.690 | 256 | 100% |
| 512 | 0.770 | 497 | 85% |
| 1024 | 0.810 | 823 | 38% |
| 2048 | 0.780 | 1131 | 23% |

Accuracy rises with budget from 0.61 to about 0.8. It is not monotone. With 100 questions the standard error is
about 4 points, so small dips such as 64 to 128 tokens and 1024 to 2048 tokens are within noise. The model
answers after being cut off mid-thought, which is the salvage behaviour the post describes. Qwen3's report shows
smooth curves for the flagship model on harder benchmarks. This run only agrees with the broad trend.

**s1-style "Wait"** (`results/04_waits.md`, 100 questions, cap 2,048 thinking tokens)

| Waits forced | Accuracy | Mean thinking tokens | Hit the cap |
|---|---|---|---|
| 0 | 0.790 | 1104 | 16% |
| 1 | 0.810 | 1312 | 29% |
| 2 | 0.750 | 1666 | 55% |

Forcing a Wait makes the model think longer (1104 to 1666 tokens) but does not improve accuracy here. Relative to the
zero-Wait run, 6 questions turned right and 4 turned wrong with one Wait, and 5 turned right and 9 turned wrong with
two. Each row samples afresh, so these flips include sampling noise. s1 reports a gain after fine-tuning a 32B model
on 1,000 reasoning traces. This lab uses an untuned 0.6B model, so it checks that the mechanism runs, not the paper's
result. More traces also hit the 2,048-token cap as Waits are added, and a cut-off trace is answered from partial work.

## Which script covers which part of the post

| Post section | Script or module |
|---|---|
| One latent variable, estimator view of a single chain | `vote.py`, `03_vote_theory.py` |
| What a token buys: depth and parity | `parity.py`, `01_parity_depth.py` |
| The bill: FLOPs, KV cache, latency | `cost.py`, `02_cost_of_thought.py` |
| Reasoning as search: voting (the Vote part) | `vote.py`, `03_vote_theory.py`, `04_qwen3_runs.py --part vote` |
| When thought stops being text: bits per step | `02_cost_of_thought.py` (log2 V only) |
| Budgets and mode switching: template, hard and soft off, budget | `budget.py`, `04_qwen3_runs.py --part modes budget` |
| Budget forcing with "Wait" | `budget.py`, `04_qwen3_runs.py --part waits` |

## What was not measured

- Large-scale results. The circuit-complexity theorem, Qwen3's training pipeline and benchmark numbers, s1's gains
  and the o1 and R1 results are the papers' own. Nothing here reproduces them. The runs use a 0.6B model and small
  Transformers, so only the mechanisms are shown.
- Tree of Thoughts, Self-Refine, Coconut, Quiet-STaR, the faithfulness experiments and the Illusion of Thinking
  puzzles are not implemented.
- Parity uses one seed per cell. The model-level runs use 100 or 200 questions and one sampling run, with no
  confidence intervals.
- A 32B model was never run. Section 2 is arithmetic on its config.
- Timings are not benchmarks. The `seconds` fields are wall-clock for an unoptimised batched Hugging Face loop with
  padding, and say nothing about serving speed.
- The decode-latency floor ignores KV-cache reads and kernel overheads.
