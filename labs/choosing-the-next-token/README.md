# Lab: Choosing the Next Token

A runnable companion to the theory post
[Choosing the Next Token](https://srini-anirudh.github.io/blogs/choosing-the-next-token/).

The post argues that the decoder is a separate program from the model. The network gives a vector of logits. The decoder
reshapes it, picks a token, and repeats. This lab writes each of those decoders from scratch in small PyTorch files and
runs them on real models, so you can watch the post's claims happen.

The models are small: SmolLM2-135M, SmolLM2-135M-Instruct and SmolLM2-360M (the 135M and 360M share one tokenizer).
Qwen2.5-0.5B, a model from a different family, is used only to score generated text. Everything runs on a CPU.

## What you build

| Module | What it does |
|---|---|
| `nexttoken/dists.py` | softmax, entropy in nats and bits, effective support, the closed form dH/dT = Var[z] / T^3 |
| `nexttoken/truncation.py` | top-k, top-p, epsilon, eta, min-p and typical as six candidate-set rules on one shared sampler |
| `nexttoken/search.py` | greedy, beam search with GNMT length normalization and EOS bookkeeping, exhaustive search on a toy tree |
| `nexttoken/penalties.py` | repetition penalty (correct and naive), frequency, presence, no-repeat n-gram, logit bias |
| `nexttoken/constrained.py` | a character automaton, a per-state token mask over the real vocabulary, masking vs conditioning |
| `nexttoken/contrastive.py` | contrastive decoding (expert minus amateur) and contrastive search (degeneration penalty) |
| `nexttoken/speculative.py` | the accept/reject rule, the residual distribution, full draft-and-verify generation |
| `nexttoken/desmoothing.py` | a softmax-bottleneck toy with a known true support |
| `nexttoken/lm.py` | load a Hugging Face model, a bare cached decode loop, text scoring |
| `nexttoken/data.py`, `metrics.py` | WikiText-2 prompts, rep-4 and distinct-2, the result-file writer |

The truncation rules match Hugging Face's `TopK`, `TopP`, `MinP`, `Epsilon`, `Eta` and `Typical` logits warpers
exactly on random logits. The penalties match `RepetitionPenaltyLogitsProcessor` and `NoRepeatNGramLogitsProcessor`.
The decode loop matches `model.generate` under greedy decoding. Tests check all of these.

The coding tutorials already cover the plain loop. [C2](../../c02-hf-stack/hfstack/generation.py) writes
`generate()` out by hand, and [C4](../../c04-llm/tinyllm/sampling.py) builds temperature, top-k and top-p. This lab
adds what the post adds on top: the other four truncation rules, entropy, search, penalties, grammars, contrast and
speculation.

## Setup

From the repository root, after `pip install -e ".[dev]"`. The first run downloads the four models from the Hugging
Face Hub (about 2.3 GB) and WikiText-2.

## Run everything

```bash
# Tests. The offline set uses tiny random models and needs no download.
python -m pytest labs/choosing-the-next-token -m "not slow and not gpu and not network"
python -m pytest labs/choosing-the-next-token -m network      # real SmolLM2 models

# Scripts. Each writes results/<script name>.txt.
python labs/choosing-the-next-token/scripts/01_entropy_temperature.py
python labs/choosing-the-next-token/scripts/02_greedy_and_beam.py
python labs/choosing-the-next-token/scripts/03_degeneration.py
python labs/choosing-the-next-token/scripts/04_truncation_zoo.py
python labs/choosing-the-next-token/scripts/05_penalties.py
python labs/choosing-the-next-token/scripts/06_constrained.py
python labs/choosing-the-next-token/scripts/07_contrastive.py
python labs/choosing-the-next-token/scripts/08_speculative.py
```

Run times on AMD EPYC 7R13, 8 threads, on a shared machine under heavy load: script 01 about 30 s; scripts 02 to 08
about 10 to 40 min each. Expect shorter times on an idle machine. The offline tests take about 20 to 30 s.

## Which script covers which section of the post

| Post section | Script |
|---|---|
| What the network actually outputs; softmax, derived | `01_entropy_temperature.py` (shift invariance), `tests/test_distributions_and_rules.py` (max entropy) |
| Entropy: the one number to hold onto | `01_entropy_temperature.py` |
| Temperature (cold and hot limits, dH/dT, the mistake it invites) | `01_entropy_temperature.py` |
| Greedy decoding and the myopia problem | `02_greedy_and_beam.py` (the post's 0.21 vs 0.36 tree) |
| Beam search, the length bias | `02_greedy_and_beam.py` |
| Why sample at all? (tail arithmetic, the repetition attractor) | `03_degeneration.py` |
| A taxonomy of truncation; order of operations | `04_truncation_zoo.py`, `03_degeneration.py` |
| Why truncation works at all (desmoothing, the bottleneck) | `04_truncation_zoo.py`, part 2 |
| Degeneration controls | `05_penalties.py` |
| Constrained decoding, and the part that is usually left out | `06_constrained.py` |
| Contrastive generation | `07_contrastive.py` |
| Three things people confuse: speculative decoding | `08_speculative.py` |

## Results

Every number below came from a run on **AMD EPYC 7R13, 8 threads, CPU only, PyTorch 2.15.0, transformers 5.22.2**.
The raw output is in `results/`, one file per script, with the hardware line and the command on top. Sampling uses
seed 0 unless the script says otherwise.

### Tests

40 tests: 38 offline (tiny random Llama models built from a config, about 20 to 30 s) and 2 `network` tests on the
real SmolLM2 models (about 40 s). All pass. There are no `slow` or `gpu` tests.

### Entropy and temperature (`01_entropy_temperature.txt`)

| Measurement | Result |
|---|---|
| dH/dT closed form vs finite difference, Zipf toy, T = 0.5 / 1 / 2 | 1.2155 vs 1.2155, 1.9327 vs 1.9327, 0.7535 vs 0.7535 |
| Odds ratio p2/p1 at T = 1 and T = 0.5 | 0.1895 and 0.0359 (the square of 0.1895) |
| SmolLM2-135M, entropy after `import numpy as` | 0.16 bits; p(` np`) = 0.989 |
| SmolLM2-135M, entropy after `The` (sentence start) | 12.47 bits, about 5700 effective options |
| 3200 WikiText-2 positions: entropy percentiles 10 / 50 / 90 | 0.90 / 4.60 / 8.24 bits |
| Adjacent positions differing by more than 3 bits | 36 % |
| T = 1 to 1.5 at a position with H = 0.5 bits vs one with H = 6 bits | 0.50 to 2.69 bits vs 6.00 to 10.02 bits |

### Greedy and beam search (`02_greedy_and_beam.txt`)

| Measurement | Result |
|---|---|
| The post's tree: greedy / beam k=2 / exhaustive best | A x, p = 0.210 / B x, p = 0.360 / B x, p = 0.360 |
| 8 English-to-French prompts, SmolLM2-135M-Instruct: mean log p of the output, greedy / beam 4 / beam 8 | -27.24 / -20.54 / -18.21 |
| Prompts where the empty output (end-of-turn first) scores higher than the output: greedy / beam 4 / beam 8 | 6 / 3 / 2 of 8 |
| Same, with GNMT length normalization (alpha = 1): beam 4 / beam 8 | 4 / 3 of 8, outputs 2 to 3 tokens longer on average |

Beam search raises log-probability over greedy, yet in 2 of 8 prompts the empty output still beats beam 8. Beam search is
not monotone in k either: on the pharmacy prompt, beam 4 (-31.71) scores below greedy (-24.46). The model is small and
its French is poor; the point is the scores, not the translations.

### Degeneration and the samplers (`03_degeneration.txt`)

The repetition attractor: a sentence repeated after a short lead-in. Mean probability SmolLM2-135M gives to the
sentence's own tokens on the next repetition, for three sentences:

| Repeats so far | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| sentence 1 | 0.075 | 0.533 | 0.831 | 0.839 | 0.846 |
| sentence 2 | 0.203 | 0.602 | 0.820 | 0.835 | 0.853 |
| sentence 3 | 0.103 | 0.631 | 0.861 | 0.867 | 0.869 |

Samplers on 8 WikiText prompts x 4 samples x 80 new tokens (greedy: 1 sample). "tail" is the fraction of emitted tokens
the model itself gave p < 0.001. NLL is under Qwen2.5-0.5B; human text scores 3.07.

| Decoder | rep-4 | distinct-2 | tail | mean set size | NLL |
|---|---|---|---|---|---|
| human reference (WikiText) | 0.008 | 0.764 | - | - | 3.07 |
| greedy | 0.268 | 0.505 | 0.000 | - | 1.28 |
| pure sampling | 0.006 | 0.867 | 0.171 | 49152 | 4.36 |
| top-k 50 | 0.013 | 0.767 | 0.006 | 50 | 2.89 |
| top-p 0.9 | 0.019 | 0.799 | 0.078 | 320 | 3.38 |
| epsilon 3e-4 | 0.004 | 0.824 | 0.058 | 167 | 3.41 |
| eta 3e-4 | 0.008 | 0.832 | 0.092 | 466 | 3.61 |
| min-p 0.1 | 0.057 | 0.587 | 0.000 | 7.4 | 2.00 |
| typical 0.9 | 0.017 | 0.799 | 0.081 | 388 | 3.44 |
| pure sampling, T = 1.5 | 0.000 | 0.997 | 0.780 | 49152 | 9.53 |
| top-p 0.9, T = 1.5, temper first | 0.000 | 0.994 | 0.710 | 14417 | 9.00 |
| min-p 0.1, T = 1.5, truncate first | 0.039 | 0.634 | 0.000 | 7.2 | 2.23 |
| min-p 0.1, T = 1.5, temper first | 0.010 | 0.788 | 0.001 | 22.6 | 2.94 |

Greedy loops (rep-4 0.268). Pure sampling does not loop, but 17 % of its tokens come from the model's own p < 0.001
tail. At T = 1.5, top-p computed on the tempered distribution admits about 14000 tokens and the text falls apart
(NLL 9.00). Min-p at the same temperature stays coherent (NLL 2.94) because its threshold scales with the top token.
This is one small run on one model; it shows the mechanism the min-p paper describes, not its benchmark results.

### The truncation zoo (`04_truncation_zoo.txt`)

Median candidate-set size by entropy bin, 3200 WikiText-2 positions, SmolLM2-135M:

| Rule | H < 0.5 bits | 2 to 4 bits | 6 to 8 bits | H > 8 bits | Spearman(H, size) | Real next token kept |
|---|---|---|---|---|---|---|
| top-k 50 | 50 | 50 | 50 | 50 | 0.03 | 0.829 |
| top-p 0.9 | 1 | 11 | 370 | 1311 | 0.97 | 0.907 |
| epsilon 3e-4 | 5 | 60 | 250 | 405 | 0.97 | 0.917 |
| eta 3e-4 | 5 | 60 | 403 | 1901 | 0.97 | 0.944 |
| min-p 0.1 | 1 | 3 | 9 | 19 | 0.76 | 0.653 |
| typical 0.9 | 1 | 11 | 372 | 1549 | 0.97 | 0.909 |

Top-k is the only rule whose set size ignores entropy. Epsilon grows slowly at high entropy: its fixed floor removes many individually rare
tokens. Eta keeps growing, because its floor relaxes as entropy rises. Min-p 0.1 keeps small sets
everywhere and drops the real next token at 35 % of positions.

| Measurement | Result |
|---|---|
| After `import numpy as` (0.16 bits): set size top-p / epsilon / eta / min-p / typical | 1 / 6 / 6 / 1 / 1 |
| Min-p 0.1, mean set size at T = 3, truncate-then-temper vs temper-then-truncate | 8.2 vs 375.6 |
| Desmoothing toy, bottleneck d = 8: mean mass outside the true support | 0.230 |
| Same toy, d = 8: recall on broad contexts (support >= 100), epsilon 1e-3 vs eta 1e-3 | 0.270 vs 0.858 |
| Same toy, d = 8: precision on sharp contexts (support <= 5), top-k 20 vs eta 1e-3 | 0.130 vs 0.948 |
| Same toy, d = 32 and d = 128: mass outside the support | 0.0001 |

### Penalties (`05_penalties.txt`)

| Measurement | Result |
|---|---|
| Sign trap: p of a seen token with logit -1, no penalty / naive divide by 1.5 / CTRL branch | 0.0389 / 0.0535 / 0.0240 |
| Add 5.0 to every logit before the penalty: p(token 1) under repetition 1.5 / under frequency 0.7 | 0.319 becomes 0.631 / 0.348 stays 0.348 |
| Greedy, 10 WikiText prompts, 100 tokens, mean rep-4: none / repetition 1.2 / frequency 0.5 / presence 0.5 / no-repeat 3-gram | 0.342 / 0.011 / 0.137 / 0.241 / 0.000 |
| Code completion, identifier reuse in prompt 1 (`total`, `values`, `value`): none / repetition 1.2 / no-repeat 3-gram | 20 / 3 / 7 |

The penalty that best breaks prose loops (repetition 1.2) is also the one that most suppresses legitimate reuse of
variable names in code. On this model the naive and CTRL forms of the repetition penalty gave identical greedy output:
the sign trap only moves tokens with negative logits, and those rarely compete for the argmax.

### Constrained decoding (`06_constrained.txt`)

| Measurement | Result |
|---|---|
| 12 extraction prompts, exact match of `{"name": "...", "age": N}`: unconstrained greedy / constrained greedy | 0 / 12 vs 12 / 12 |
| Same, parseable JSON with both keys anywhere in the reply | 12 / 12 vs 12 / 12 |
| Same, name and age both correct: greedy unconstrained / constrained; T = 1 unconstrained / constrained | 11 / 12 vs 12 / 12; 8 / 12 vs 10 / 12 |
| Automaton states indexed; mean allowed tokens per state | 24; 21536 of 49152 |
| Exact TV(masked sampler, p(y \| y in L)), 5 random toy models | 0.290 to 0.479 |
| After importance resampling with 64 particles | 0.010 to 0.020 (sampling noise is about 0.01 to 0.02) |

Unconstrained, the model writes pretty-printed JSON, so it never matches the compact pattern, yet it is parseable every
time under greedy. The mask fixes the format. It does not fix the content: with sampling, 2 of 12 constrained records
still have a wrong name or age (for example `"name": "Maria"`).

### Contrastive methods (`07_contrastive.txt`)

8 WikiText prompts, 100 new tokens. Expert SmolLM2-360M, amateur SmolLM2-135M.

| Decoder | rep-4 | distinct-2 | NLL (Qwen2.5-0.5B) |
|---|---|---|---|
| human reference | 0.012 | 0.712 | 3.05 |
| greedy, 360M | 0.409 | 0.398 | 0.91 |
| top-p 0.95, 360M | 0.006 | 0.867 | 3.22 |
| contrastive decoding, alpha 0.1 | 0.050 | 0.799 | 2.52 |
| contrastive decoding, no plausibility constraint | 0.034 | 0.649 | 6.20 |
| contrastive search, k = 5, a = 0.6 | 0.101 | 0.668 | 1.60 |

Without the plausibility constraint, the top tokens by log-ratio have expert probabilities near 1e-7 (`ocument`,
`airobi`), and the output is junk. With alpha = 0.1 they are ` church`, ` narrow`, ` bridge`.

### Speculative sampling (`08_speculative.txt`)

| Measurement | Result |
|---|---|
| Synthetic p, q, 200000 draws: TV(speculative, p) vs TV(iid draws from p, p) | 0.0025 to 0.0052 vs 0.0019 to 0.0052 |
| Synthetic: measured acceptance rate vs 1 - TV(q, p) | 0.938 vs 0.939, 0.263 vs 0.262 (and three more cases) |
| Real models, first token, 4000 rounds: TV(draft, target) / TV(speculative, target) / sampling noise | 0.072 / 0.0075 / 0.0146 |
| Greedy speculative vs target greedy, 6 prompts x 64 tokens | 6 / 6 identical; 110 target passes instead of 384 |
| T = 1, gamma 1 / 2 / 4 / 6: draft acceptance | 0.789 / 0.652 / 0.451 / 0.356 |
| T = 1, gamma 1 / 2 / 4 / 6: tokens per target pass | 1.78 / 2.26 / 2.76 / 3.00 |

The output distribution is the target's, within sampling noise, even though the draft differs from it by TV 0.072.

## What was not measured

- No paper's scale is reproduced. The models have 135M and 360M parameters, and most comparisons use 8 to 12 prompts.
  Treat the sampler tables as a demonstration of each mechanism, not a ranking of samplers.
- No human evaluation. NLL under Qwen2.5-0.5B is a proxy, and very low NLL usually means dull or looping text.
- Stahlberg and Byrne's exact search is not run. `02_greedy_and_beam.py` only compares beam outputs with the empty
  output.
- Hewitt et al.'s "Donald" example is not reproduced. SmolLM2-135M does not put most of its mass on one token after
  "Donald" (12.2 bits of entropy), so top-p does not collapse there. The collapse shows up after `import numpy as`.
- The desmoothing toy has independent contexts, so its smoothing comes only from the bottleneck. The parameter sharing
  of a real model is not modelled. With d = 32 or more the toy fits its finite sample and leaks almost nothing.
- The basis-aware sampler of Finlayson et al., DoLa, diverse beam search, Mirostat, sequential Monte Carlo over a real
  model, Medusa and EAGLE are not built.
- Tam et al.'s reasoning decline under format restrictions is not tested. `06_constrained.py` uses an extraction task.
- `lm.load` reads `NEXTTOKEN_DEVICE` (default `cpu`) so the model forwards could run on a GPU. That path was never
  run, and the contrastive and speculative modules call the models with CPU tensors.
- Speculative decoding is measured in target forward passes only. Wall-clock speed is not measured: the post's speed
  argument depends on a memory-bound GPU, and no GPU was used.
