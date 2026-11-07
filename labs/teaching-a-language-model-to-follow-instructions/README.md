# Lab: Teaching a Language Model to Follow Instructions

Code companion for the post
[Teaching a Language Model to Follow Instructions](https://srini-anirudh.github.io/blogs/teaching-a-language-model-to-follow-instructions/).
The post argues that supervised fine-tuning (SFT) keeps the pretraining loss, swaps the data distribution, and chooses
which tokens are charged. This lab builds each of those steps in small Python and measures them on real models.

What you can run:

- Chat templates of three real tokenizers (SmolLM2, Qwen2.5, Llama 3): token ids, special tokens, the generation
  prompt, a doubled BOS, forged role markers, and the exact assistant-only loss mask.
- The mask proved three ways: a manual masked sum, the Hugging Face `labels=-100` loss, and `-log p(prompt + response)`
  minus `-log p(prompt)` with no mask at all. Also: masked tokens are still attended to.
- Multi-turn examples (one pass against one example per turn), packing without a document mask, and what truncation
  does to the end-of-turn token.
- Fine-tuning SmolLM2-135M with assistant-only, full-sequence and instruction-modelling losses in two data regimes,
  with several seeds.
- Behaviour cloning on a toy: compounding error, recovery data and DAgger.
- Format adherence against the number of SFT examples, against few-shot prompting of the same base model.
- A small version of the Gekhman et al. known-versus-unknown experiment on TriviaQA.
- Rejection sampling on GSM8K, and how many answer-correct solutions contain a false equation.
- A task where a written trace buys serial depth, with an exact step checker for the outcome filter.
- Learning rate, epochs, truncation, verbose data, a benchmark "drop" that is a format problem, and a template
  mismatch between training and serving.

Every dataset and model is public and downloads without a login: Alpaca-cleaned, Dolly-15k, GSM8K, TriviaQA,
ARC-Easy, WikiText-2, SmolLM2-135M and 360M (base and Instruct), Qwen2.5-0.5B-Instruct, and the tokenizer of a Llama 3
copy.

## Layout

```
labs/teaching-a-language-model-to-follow-instructions/
├── sftlab/
│   ├── templates.py   # render, assistant spans, Encoded examples with per-token loss weights, four mask modes
│   ├── checks.py      # the loss three ways, exploded multi-turn examples, gradients, packed rows
│   ├── train.py       # SFT loop with an explicit weight per token; per-token and per-example normalisation
│   ├── lm.py          # model loading, per-token NLL, batched greedy and sampled generation
│   ├── pipeline.py    # encode lists of examples, fine-tune a fresh copy, behaviour probes, serving variants
│   ├── data.py        # Alpaca-cleaned, Dolly-15k, seeded splits
│   ├── formats.py     # five trained format constraints and one held-out, each with an exact checker
│   ├── divergence.py  # entropy, cross-entropy, KL, Gaussian fits under forward and reverse KL
│   ├── bc.py          # ring task: behaviour cloning, noisy execution, DAgger; the worst-case tabular chain
│   ├── chain.py       # running sum mod 10: direct target, trace target, step checker
│   ├── arith.py       # finds 'a op b = c' statements in free text and checks them
│   ├── verifiers.py   # JSON, label, math, code checkers; capped mixing; ROUGE-L; n-gram contamination
│   ├── metrics.py     # token F1, distinct-n
│   └── toy.py         # a byte-level tokenizer with a ChatML template and a tiny Llama, for offline tests
├── scripts/           # 01 to 10, one per post section, listed below
├── results/           # real outputs; the first lines of each file give the hardware and the command
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. The first run downloads the models (about 3 GB) and the
datasets from the Hugging Face Hub. PyTorch uses every core by default, so set `OMP_NUM_THREADS` on a shared machine.
The loader picks CUDA if it is visible and the CPU otherwise; set `SFTLAB_DEVICE=cpu` to force the CPU.

## Run everything

```bash
python labs/teaching-a-language-model-to-follow-instructions/scripts/01_the_gap.py            # GPU or CPU, about 2 min on GPU
python labs/teaching-a-language-model-to-follow-instructions/scripts/02_same_loss.py          # GPU, about 5 min
python labs/teaching-a-language-model-to-follow-instructions/scripts/03_behaviour_cloning.py  # CPU, about 3 min
python labs/teaching-a-language-model-to-follow-instructions/scripts/04_anatomy.py            # CPU, about 12 min
python labs/teaching-a-language-model-to-follow-instructions/scripts/05_loss_mask.py          # GPU, about 45 min
python labs/teaching-a-language-model-to-follow-instructions/scripts/06_what_sft_teaches.py   # GPU, about 1 h
python labs/teaching-a-language-model-to-follow-instructions/scripts/07_data_taxonomy.py      # CPU, about 2 min
python labs/teaching-a-language-model-to-follow-instructions/scripts/08_data_construction.py  # GPU, about 7 min
python labs/teaching-a-language-model-to-follow-instructions/scripts/09_reasoning_sft.py      # GPU, about 15 min
python labs/teaching-a-language-model-to-follow-instructions/scripts/10_how_much_sft.py       # GPU, about 1 h 10 min
```

Training and generation scripts used one H100 each, on a shared machine, so the times are a guide. Training is slower
than the model size suggests (about 0.3 s per step at batch 16 and up to 384 tokens), most of it fp32 logits over a
49,152-token vocabulary. A CPU run of scripts 05, 06, 09 and 10 would take many hours; pass fewer `--seeds`, `--Ns` or
`--parts` to try them. Do not compare CPU numbers with the tables below.

## Test

```bash
python -m pytest labs/teaching-a-language-model-to-follow-instructions -m "not slow and not gpu and not network"   # 22 passed, about 20 s
python -m pytest labs/teaching-a-language-model-to-follow-instructions -m network                                  # 4 passed
```

The offline tests use a byte-level tokenizer with a ChatML template and a 2-layer random Llama. They check the mask on
every mode (assistant, last turn, all, instruction modelling), that the prompt is a prefix of the full sequence, that
truncation removes the turn-end token, that the three ways of computing the loss agree, that masked tokens change the
loss of the tokens they precede, that one pass over three turns has the same gradient as three exploded examples, that a
document mask makes packing exact, per-token against per-example normalisation, the divergence identity and the
forward/reverse fits, the worst-case chain's two slopes, every format checker, the step checker, the arithmetic parser,
and the data checkers. The `network` tests check the real SmolLM2 and Qwen templates, forged role markers, the Llama 3
doubled BOS, and the datasets.

## Results

Hardware: scripts 03, 04 and 07 ran on **AMD EPYC 7R13, 8 threads, torch 2.15.0, CPU only**. Scripts 01, 02, 05, 06, 08,
09 and 10 ran on **NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2**, one GPU at a time, bf16 autocast for training.
Each result file starts with its own hardware line and command. A seed fixes the data subset and the data order; the
pretrained weights are the same in every run. "Single seed" means one run.

### The gap a base model leaves (`01_the_gap.py`)

SmolLM2-135M (base) and SmolLM2-135M-Instruct, greedy, 96 new tokens. The base model gets either the bare question or the
question inside the ChatML template it never saw.

| Measurement | Base, bare question | Base, chat template | Instruct, chat template |
|---|---|---|---|
| Top next token after "What is the capital of France?" | newline, 0.695 | "What", 0.368 | "The", 0.831 |
| Entropy of that distribution (bits) | 2.74 | 6.14 | 1.50 |
| Emitted a stop token (12 short factual questions) | 0.00 | 0.00 | 0.92 |
| Emitted a stop token (112 prompts, 96-token budget) | 0.00 | 0.01 | 0.29 |
| Reply repeats the question | 0.13 | 0.67 | 0.00 |
| Reply contains a new question mark | 0.36 | 0.33 | 0.01 |
| Gold answer in the first 20 tokens (12 factual questions) | 0.83 | 0.00 | 0.83 |

On a bare question this base model does answer: a newline first, then "The capital of France is Paris.", then it repeats
the question and the answer until the budget runs out. What it lacks is a stop and a turn boundary. Inside the chat
template it has never seen, it echoes the question and emits junk tokens (`pite`, `ostering`). The Instruct model's low
overall stop rate (0.29) is the budget: many Alpaca answers are longer than 96 tokens.

### Same loss, different distribution (`02_same_loss.py`)

| Check | Result |
|---|---|
| Cross-entropy = H(D) + KL(D‖p), six random distributions over 50 symbols | the two sides agree to four decimals in every row (for example 6.0996 and 6.0996); a 10^6-draw Monte Carlo gives 6.0995 |
| One Gaussian fitted to 0.72 N(-3, 0.6) + 0.28 N(3, 0.6), forward KL | mean -1.30, sd 2.75; 26% of its mass sits between the bumps where D has none |
| The same fit under reverse KL | mean -3.00, sd 0.60; all mass on the heavier bump, none on the other |
| D = (0.6, 0.3, 0.1, 0): p puts 0.20 on a behaviour D never shows | KL(D‖p) = 0.223, KL(p‖D) = infinite |
| p puts 0.01 on a behaviour D shows 10% of the time | KL(D‖p) = 0.230 |
| p drops the 10% behaviour | KL(D‖p) = infinite, KL(p‖D) = 0.105 |

NLL per token in nats, 200 documents each, same tokenizer and same function:

| Model | WikiText paragraphs | Chat text, every token | Chat text, assistant tokens |
|---|---|---|---|
| SmolLM2-135M (base) | 3.139 | 4.151 | 2.349 |
| SmolLM2-135M-Instruct | 3.279 | 2.722 | 1.630 |

SFT on 800 Alpaca examples where 80% of replies end with a sign-off nobody asked for ("Hope this helps."), 3 seeds:

| Seed | Share of training replies with the sign-off | Held-out replies with it, greedy | Held-out replies with it, sampled at T = 1 |
|---|---|---|---|
| 0 | 0.806 | 0.994 | 0.609 |
| 1 | 0.785 | 0.989 | 0.657 |
| 2 | 0.802 | 1.000 | 0.756 |
| Mean | 0.798 | 0.994 | 0.674 |

Greedy decoding picks the dominant style. Sampling keeps the minority style, and more of it than the data has. The loss
has no term that separates a style you want from one you do not.

### SFT is behaviour cloning (`03_behaviour_cloning.py`)

Part A is the classical construction: per-step error eps on expert states and no way back. It is exact by construction.
Part B is learned: an MLP cloned from expert (state, action) pairs on a ring task, where the expert's states hug the ring
within 0.037. Cost is the number of steps spent more than 0.1 from the ring. Five seeds; mean cost with the range over seeds.

| Check | Result |
|---|---|
| Worst-case chain, eps = 0.001: log-log slope of cost against T (T = 10 to 200) | 1.95 without recovery, 0.99 with recovery |
| Same, eps = 0.001, T = 100 | simulated 4.946, exact sum 4.887, eps T^2 / 2 = 5.0; with recovery 0.100 = eps T |

| Policy | T = 25 | T = 100 | T = 400 |
|---|---|---|---|
| Expert | 0.00 | 0.00 | 0.00 |
| Behaviour cloning | 2.66 [0.0, 12.8] | 13.98 [0.0, 58.1] | 156.47 [0.6, 289.7] |
| Cloning with noisy execution, clean labels | 0.00 | 0.00 | 0.00 |
| DAgger, 1 iteration | 0.84 [0.0, 3.2] | 15.78 [0.0, 76.5] | 76.50 [0.0, 376.5] |
| DAgger, 2 iterations | 0.00 | 0.01 | 0.04 |
| DAgger, 3 iterations | 0.06 | 0.27 | 1.35 [0.0, 6.8] |
| DAgger, 4 iterations | 0.00 | 0.00 | 0.00 |

Local log-log slopes of the cloning cost between consecutive horizons: 1.39, 1.01, 1.63, 1.86. The seed range is wide: one
seed's policy never leaves the ring and another leaves within 25 steps. The slope is not a clean test of the T^2 bound (the
bound is worst case, the cost here is capped at T, five seeds is few). What the toy does show is the mechanism: the data never
showed the way back, a rollout that leaves stays out, and recovery data or on-policy labels remove it.

### Anatomy of an SFT example (`04_anatomy.py`)

| Check | Result |
|---|---|
| Same 3-message conversation, scored tokens of all tokens | SmolLM2 25 of 53, Qwen2.5 25 of 52, Llama 3 25 of 53; in each the last scored token is the turn-end token (`im_end` for SmolLM2 and Qwen2.5, `eot_id` for Llama 3) |
| Tokenise the rendered text with the default `add_special_tokens=True` | SmolLM2 and Qwen2.5: no change. Llama 3: ids start `[128000, 128000, ...]`, a doubled BOS |
| User message containing `im_end` and `im_start` followed by `system` as text, default tokenisation | the typed strings become the special tokens: `im_start` x3 and `im_end` x3, against x2 each for an honest message |
| The same content tokenised alone with `split_special_tokens=True` | zero special ids |
| Three assistant turns: one pass against three exploded examples, gradient of the summed loss | relative difference 1.5e-6 (fp32); the exploded examples push 189 tokens through the model against 89 (2.12x) |
| "Score only the final turn" | 16 of 23 assistant tokens (70%) |

Serving the released SmolLM2-135M-Instruct with a small template mistake (100 held-out Alpaca prompts, greedy, 96 new tokens):

| Serving format | NLL per token of the reference reply | Emitted a stop token | First 30 characters differ from the correct template's reply |
|---|---|---|---|
| Correct template | 1.646 | 0.52 | 0.00 |
| No newline after `assistant` | 2.155 | 0.52 | 1.00 |
| Extra blank line after the header | 1.881 | 0.44 | 0.95 |
| No system turn | 1.671 | 0.43 | 0.54 |
| No newline after the user turn | 1.671 | 0.69 | 0.69 |
| Plain `User:` / `Assistant:` text | 1.924 | 0.67 | 1.00 |

Packing two chat examples into one row, SmolLM2-135M base, fp32, 40 pairs; assistant-token NLL of the second example minus
its NLL alone:

| Packing | Mean difference | Mean absolute difference | Largest |
|---|---|---|---|
| Causal mask only | -0.361 | 0.424 | 1.281 |
| Causal mask, positions reset | +1.273 | 1.290 | 3.675 |
| Document mask | 0.0000 | 0.0000 | 0.0000 |
| Document mask, positions reset | 0.0000 | 0.0000 | 0.0000 |

Truncation (3,000 random examples per dataset, whole-conversation token length):

| Dataset | max length | Examples cut | Cut examples that lose `im_end` | Assistant tokens lost |
|---|---|---|---|---|
| Alpaca | 64 | 0.877 | 0.990 | 0.890 |
| Alpaca | 256 | 0.307 | 0.995 | 0.227 |
| Alpaca | 512 | 0.018 | 0.981 | 0.009 |
| Dolly | 64 | 0.836 | 0.996 | 0.852 |
| Dolly | 256 | 0.242 | 0.992 | 0.306 |
| Dolly | 512 | 0.068 | 0.985 | 0.122 |

### Which tokens receive loss (`05_loss_mask.py`)

| Check | Result |
|---|---|
| Assistant-only sum of NLL, 20 examples, three ways (manual mask; Hugging Face `labels=-100`; -log p(prompt + response) minus -log p(prompt), no mask) | largest absolute difference 3.6e-5 between the first two, 1.8e-4 between the first and third (largest relative difference 2.7e-6), fp32 |
| One reply, assistant-token NLL after its own prompt / after a different prompt / with its own prompt hidden from attention | 3.381 / 3.529 / 4.074: masked tokens are still context |
| Share of tokens scored, Alpaca / Dolly | full sequence 0.995 / 0.995; assistant only 0.737 / 0.382; instruction modelling (w = 0.3) 0.767 / 0.518 |
| Default system prompt | 21 tokens, 10.8% of an average Alpaca example and 9.9% of a Dolly one |
| Loss weight on the 4 longest of 16 Dolly examples | 61.1% under per-token normalisation, 25% under per-example |

Fine-tuning SmolLM2-135M (base), lr 2e-5, batch 16, three seeds (mean +- std over seeds; each seed also changes the
validation split). "Token NLL" is the mean over scored validation tokens, "example NLL" the mean over examples.
Validation is always the assistant tokens of held-out replies.

| Regime | Loss | Token NLL | Example NLL | Token F1 | Stopped |
|---|---|---|---|---|---|
| R1: Alpaca, N = 1000, 3 epochs | assistant only | 1.592 +- 0.026 | 1.681 +- 0.044 | | |
| | full sequence | 1.605 +- 0.026 | 1.689 +- 0.039 | | |
| | instruction modelling, w = 0.3 | 1.593 +- 0.026 | 1.680 +- 0.043 | | |
| | assistant only, per-example normalisation | 1.615 +- 0.027 | 1.665 +- 0.044 | | |
| R2a: Dolly, long instruction and short reply, N = 100, 10 epochs | assistant only | 1.334 +- 0.044 | 1.610 +- 0.023 | 0.261 | 0.12 |
| | full sequence | 1.533 +- 0.016 | 1.911 +- 0.076 | 0.206 | 0.00 |
| | instruction modelling, w = 0.3 | 1.382 +- 0.034 | 1.684 +- 0.055 | 0.237 | 0.01 |
| R2b: the same pool, N = 800, 3 epochs | assistant only | 1.174 +- 0.044 | 1.340 +- 0.011 | 0.372 | 0.78 |
| | full sequence | 1.299 +- 0.034 | 1.541 +- 0.037 | 0.232 | 0.18 |
| | instruction modelling, w = 0.3 | 1.216 +- 0.044 | 1.401 +- 0.008 | 0.359 | 0.72 |
| | assistant only, per-example normalisation | 1.192 +- 0.046 | 1.317 +- 0.012 | 0.363 | 0.80 |

The R2 pool is the 2,228 Dolly examples with a context over 400 characters and a reply under 200 characters (closed QA,
information extraction, summarisation). Without fine-tuning the same validation set scores token NLL 2.481. The F1 is word
overlap with the reference, "Stopped" the share of greedy replies that emitted `im_end` within 64 tokens.

### What does SFT actually teach (`06_what_sft_teaches.py`)

**A. Format adherence against N.** SmolLM2-135M (base), Alpaca prompts, five trained format constraints (lowercase, a fixed
sign-off, one sentence, exactly three `- ` lines, wrapped in quotes) and one never trained ("Begin your answer with the
word Sure,"). Evaluation: 40 held-out prompts x 7 = 280 prompts, greedy, 160 tokens. A reply "adheres" only if it ended
with the turn-end token inside the budget and passes the exact checker. Training: lr 5e-5, batch 16, about 150 optimiser
steps for every N, so N is the only thing that changes (small N is repeated many times). Three seeds, mean +- std.

| N | Epochs | Adheres, trained constraints | Adheres, unseen constraint | Stops, no constraint |
|---|---|---|---|---|
| 0 (base, chat template) | - | 0.000 | 0.000 | 0.000 (single run) |
| 16 | 150 | 0.430 +- 0.040 | 0.375 | 0.892 |
| 64 | 38 | 0.540 +- 0.027 | 0.375 | 0.867 |
| 256 | 10 | 0.672 +- 0.022 | 0.358 | 0.742 |
| 1024 | 3 | 0.742 +- 0.030 | 0.300 | 0.675 |
| Few-shot prompting, 5 demonstrations, base model | - | 0.185 +- 0.058 | 0.000 | 0.917 |
| Few-shot prompting, 10 demonstrations, base model | - | 0.183 +- 0.039 | 0.000 | 0.758 |

Sixteen examples are enough to turn "never stops" into "usually stops" (0.89) and to reach 0.43 on exact formats. More
examples keep helping the trained formats. The unseen format does not follow the same curve: 0.375 at N = 16 and 0.300
at N = 1024. Few-shot prompting of the same base model, in a plain `### Instruction` text format with the reply cut at the
next `###`, reached 0.18.

**B. Known against unknown answers** (a small version of Gekhman et al.). SmolLM2-360M base, 5,995 TriviaQA validation
questions, closed book. 1,910 (31.9%) are answered correctly with a 5-shot prompt: these are "known". Each run fine-tunes
on 400 questions with a given share of unknown answers, 8 epochs, lr 2e-5. **Single seed per row.** Fit = accuracy on
150 training questions of that kind; dev = 150 known and 150 unknown questions never trained on (one dev question moves
accuracy by 0.007; the sampling error of 150 questions is about 0.04).

| Unknown share | Epoch | Fit: known | Fit: unknown | Dev: known | Dev: unknown |
|---|---|---|---|---|---|
| 0% | 1 | 0.713 | - | 0.560 | 0.047 |
| 0% | 3 | 0.907 | - | 0.533 | 0.047 |
| 0% | 8 | 1.000 | - | 0.480 | 0.053 |
| 50% | 1 | 0.633 | 0.107 | 0.587 | 0.060 |
| 50% | 3 | 0.873 | 0.480 | 0.480 | 0.060 |
| 50% | 8 | 1.000 | 0.967 | 0.393 | 0.040 |
| 100% | 1 | - | 0.107 | 0.540 | 0.080 |
| 100% | 3 | - | 0.400 | 0.480 | 0.047 |
| 100% | 8 | - | 0.927 | 0.373 | 0.053 |

Unknown answers are fitted more slowly than known ones (at epoch 3 in the 50% run: 0.480 against 0.873), and the accuracy on
questions the model knew falls as unknown answers are absorbed: after 8 epochs 0.480, 0.393 and 0.373 for 0%, 50% and 100%
unknown. The gaps between the rows are about two standard errors with one seed, so this is suggestive, not established.
Accuracy on the unknown dev questions stays near 0.05 in every run: nothing new was learned that generalises. The known
dev set starts at 1.0 by construction under the 5-shot prompt and at 0.54 to 0.59 after one epoch in the chat format, so
part of every drop is the change of prompt format, which the 0% row isolates.

### The instruction-data taxonomy and how it gets built (`07_data_taxonomy.py`, CPU)

Median token counts with the SmolLM2 tokenizer (1,500 random examples per row; "ratio" is prompt-plus-context over reply):

| Set | Examples | Prompt tokens | Reply tokens | Ratio |
|---|---|---|---|---|
| Dolly open QA | 3,742 | 10 | 35 | 0.29 |
| Dolly general QA | 2,191 | 10 | 92 | 0.11 |
| Dolly classification | 2,136 | 27 | 27 | 1.00 |
| Dolly closed QA | 1,773 | 197 | 26 | 7.58 |
| Dolly brainstorming | 1,766 | 12 | 57 | 0.21 |
| Dolly information extraction | 1,506 | 223 | 31 | 7.19 |
| Dolly summarisation | 1,188 | 214 | 75 | 2.85 |
| Dolly creative writing | 709 | 14 | 146 | 0.10 |
| Alpaca-cleaned | 51,760 | 14 | 107 | 0.13 |

Closed QA, information extraction and summarisation are the "long instruction, short reply" categories used for regime R2 above.

| Check | Result |
|---|---|
| Exact checkers for JSON, a closed label set, a numeric answer and code with asserts | each accepts a correct reply and rejects a wrong one |
| Token F1 of a Dolly summary against a copy with numbers changed and negations inserted (300 summaries) | 0.948, against 0.126 for another example's summary: overlap cannot see the error |
| Proportional mixing with one 900k source and one 1k source (sizes illustrative) | 92.3% and 0.1% of the mix; a 20,000 cap per source gives 31.5% and 1.6% |
| GSM8K test questions (1,319) sharing a word n-gram with Alpaca + Dolly (66,771 examples) | n = 13: 0; n = 8: 3 questions (0.0023); n = 5: 419 (0.318) |
| Planted exact copy of 100 test questions | detected 100/100 at n = 13, 8 and 5 |
| Light paraphrase (every 5th word replaced) of the same 100 | detected 0/100 at n = 13 and n = 8, 36/100 at n = 5; ROUGE-L against the original 0.81 |
| Alpaca-cleaned, 5,000 random instructions: share with ROUGE-L >= 0.7 to an earlier one | 0.002 (0.432 at >= 0.5, 0.925 at >= 0.3; lower bounds) |

The Alpaca figure is low because Alpaca was itself built with the Self-Instruct pipeline, whose filter drops instructions
at that threshold. It says the filter did its job on this set, not what an unfiltered bootstrap would produce.

### How the data gets built: rejection sampling (`08_data_construction.py`, single seed)

Qwen2.5-0.5B-Instruct samples k = 16 solutions for each of 200 GSM8K training questions at temperature 0.8 (up to 320
tokens). A solution is "kept" if its last number equals the gold answer (the outcome filter). The check for a false equation
is a parser for plain `a op b = c` statements. **Single seed; GPU sampling is not bit-reproducible, so a rerun differs a little.**

| k | pass@k | Kept per question | Distinct paths kept per question | Share of kept that repeat a path | Kept with a false equation |
|---|---|---|---|---|---|
| 1 | 0.290 | 0.29 | 0.29 | 0.000 | 0.017 |
| 2 | 0.425 | 0.60 | 0.52 | 0.142 | 0.008 |
| 4 | 0.550 | 1.18 | 0.80 | 0.323 | 0.009 |
| 8 | 0.635 | 2.35 | 1.35 | 0.424 | 0.011 |
| 16 | 0.730 | 4.69 | 2.06 | 0.561 | 0.007 |

Single-sample accuracy is 0.293. A false equation appears in 5.0% of all samples, 0.7% of the answer-correct ones and 6.8%
of the answer-wrong ones. Dropping answer-correct solutions with a false equation removes 7 of 938 (0.7%). The kept set grows
with k, but more than half of it at k = 16 repeats a path already kept. The false-equation share among kept solutions rests on
fewer than 10 solutions per row; across earlier runs of the same script the k = 1 value ranged from 0% to 5% and the k = 16
value from 0.5% to 1.6%. The parser also misreads some algebra, so the number only says that answer-correct solutions with a
wrong step can exist.

### Reasoning SFT (`09_reasoning_sft.py`, single seed)

SmolLM2-135M base fine-tuned two ways on one task: the sum of n digits modulo 10, n from 2 to 12 in training (12,000
examples, 2 epochs, lr 1e-4). The direct target is "Answer: d". The trace target writes the running total after every digit,
then the answer. 300 fresh questions per n, greedy; chance for the final answer is 0.10. **Single seed.**

| n | Direct answer | With trace | Trace model: per-step accuracy |
|---|---|---|---|
| 2 | 1.000 | 1.000 | 1.0000 |
| 4 | 0.617 | 1.000 | 1.0000 |
| 6 | 0.187 | 1.000 | 1.0000 |
| 8 | 0.133 | 1.000 | 1.0000 |
| 10 | 0.103 | 1.000 | 1.0000 |
| 12 | 0.100 | 1.000 | 1.0000 |
| 16 (longer than any training example) | 0.107 | 0.000 | 0.0088 |
| 20 (longer than any training example) | 0.120 | 0.000 | 0.0000 |

The direct model reaches chance by n = 8 to 10; the trace model is exact through n = 12. Neither extends beyond the training
lengths: the trace model breaks its format at n = 16 and 20.

For the outcome filter the fully trained model never slips, so the same trace model was trained on fewer examples (1 epoch,
lr 1e-4):

| Training examples | Chain accuracy at n = 12 (greedy) | Per-step accuracy | (per-step)^12 |
|---|---|---|---|
| 600 | 0.080 | 0.1825 | 0.000 |
| 1,200 | 0.115 | 0.2621 | 0.000 |
| 2,400 | 0.975 | 0.9979 | 0.975 |
| 4,800 | 0.990 | 0.9992 | 0.990 |

The chain accuracy equals the product of per-step accuracies, as the post's reliability argument says. The 2,400-example
model (chosen as the one with chain accuracy nearest 0.6; the jump from 1,200 to 2,400 is sharp) was then sampled at
temperature 1.0, 16 samples for each of 200 questions with n = 12. 6.5% of all samples contain a wrong step. "Kept" means the
final digit is right; "bad" means kept with at least one wrong written step, found by an exact step checker.

| k | Kept per question | Bad among kept | Questions with a kept sample |
|---|---|---|---|
| 1 | 0.94 | 0.000 | 0.945 |
| 2 | 1.88 | 0.000 | 0.985 |
| 4 | 3.75 | 0.000 | 1.000 |
| 8 | 7.51 | 0.001 | 1.000 |
| 16 | 14.99 | 0.002 | 1.000 |

Bad kept samples are about 0.1% here, far below the 0.7% expected if a wrong trace hit the right digit by chance. In this task
a slip is carried forward, so the final digit is wrong unless a second slip cancels the first. The false-positive problem the
post describes needs a task where errors can cancel or the answer can be guessed.

### How much SFT (`10_how_much_sft.py`)

SmolLM2-135M (base) on Alpaca. "Wiki" is the NLL on 100 WikiText paragraphs (what moved elsewhere), "stop" the share of
greedy replies that ended with the turn-end token within 128 tokens, "distinct-2" the share of distinct bigrams among 8
sampled replies to the same prompt.

**A. Learning rate** (N = 1000, 3 epochs, cosine decay, single seed per row):

| lr | Val NLL | Wiki NLL | Stop | Mean length | Distinct-2 |
|---|---|---|---|---|---|
| none (base) | 1.999 | 2.944 | 0.00 | 128.0 | 0.926 |
| 1e-6 | 1.787 | 2.944 | 0.00 | 128.0 | 0.927 |
| 5e-6 | 1.659 | 2.955 | 0.00 | 128.0 | 0.872 |
| 2e-5 | 1.573 | 2.978 | 0.54 | 88.8 | 0.860 |
| 1e-4 | 1.613 | 3.125 | 0.67 | 74.0 | 0.825 |
| 5e-4 | 2.010 | 3.874 | 0.61 | 80.4 | 0.804 |

**B. Epochs on 200 examples** (constant lr 2e-5, mean of 2 seeds):

| Epochs | Train NLL | Val NLL | Wiki NLL | Greedy replies near a training reply | Distinct-2 | Mean length | Stop |
|---|---|---|---|---|---|---|---|
| 1 | 1.700 | 1.746 | 2.946 | 0.000 | 0.915 | 128.0 | 0.00 |
| 3 | 1.395 | 1.639 | 2.961 | 0.000 | 0.854 | 128.0 | 0.00 |
| 6 | 1.066 | 1.628 | 2.979 | 0.000 | 0.826 | 98.2 | 0.44 |
| 10 | 0.691 | 1.725 | 3.018 | 0.000 | 0.825 | 87.2 | 0.56 |
| 20 | 0.099 | 2.432 | 3.147 | 0.000 | 0.784 | 94.1 | 0.51 |
| 30 | 0.019 | 2.950 | 3.229 | 0.000 | 0.770 | 87.4 | 0.63 |

**C. Truncated training data** (N = 500, 3 epochs, lr 2e-5, 2 seeds; only 96 optimiser steps, so even the clean run stops rarely):

| Max length L | Examples cut | Cut examples that keep the turn-end token | Stop rate | Mean length |
|---|---|---|---|---|
| 48 | 0.99 | 0.007 | 0.000 | 128.0 |
| 96 | 0.66 | 0.006 | 0.025 | 125.2 |
| 384 | 0.01 | 0.000 | 0.180 | 113.1 |

**D. Verbose data** (N = 500 from the shorter or longer half of Alpaca replies, median 99 tokens, 2 seeds, budget 160 tokens):

| Training replies | Mean training length | Mean generated length | Stop rate |
|---|---|---|---|
| Shorter half | 40.2 | 93.4 | 0.59 |
| Longer half | 176.5 | 156.2 | 0.065 |

The generated lengths are cut by the 160-token budget, so the gap is larger than it looks.

**E. Is a post-SFT drop a format problem?** 300 ARC-Easy questions, chance 0.25. SFT on 1000 Alpaca examples, lr 2e-5, 3 epochs, 2 seeds:

| Model | Log-likelihood accuracy | Generated letter, strict | Generated, lenient |
|---|---|---|---|
| Base, plain prompt | 0.583 | 0.000 | 0.533 |
| After SFT, chat template | 0.580 | 0.000 | 0.332 |

**F. Train/serve template mismatch** on our own fine-tuned model (N = 1000, 3 epochs, lr 2e-5, mean of 2 seeds, 100 held-out prompts, 96 new tokens):

| Serving format | NLL per token of the reference reply | Emitted a stop token | First 30 characters differ | Mean new tokens |
|---|---|---|---|---|
| Correct template | 1.359 | 0.69 | 0.00 | 49.7 |
| No newline after `assistant` | 1.595 | 0.74 | 1.00 | 47.0 |
| Extra blank line after the header | 1.623 | 0.45 | 0.89 | 68.2 |
| No system turn | 1.650 | 0.01 | 0.83 | 95.7 |
| No newline after the user turn | 1.379 | 0.60 | 0.56 | 58.1 |
| Plain `User:` / `Assistant:` text | 2.292 | 0.00 | 1.00 | 96.0 |

## Post sections and scripts

| Post section | Script | What it shows |
|---|---|---|
| 1. The gap a base model leaves | `01_the_gap.py` | next-token distributions, stopping, repeating, with and without the template |
| 2. Same loss, different distribution | `02_same_loss.py` | CE = H + KL, forward and reverse KL, the same loss on web and chat text, an 80/20 style mixture |
| 3. SFT is behaviour cloning | `03_behaviour_cloning.py` | worst-case chain, a learned ring task, noisy execution, DAgger |
| 4. Anatomy of an SFT example | `04_anatomy.py` | three real templates, doubled BOS, forged markers, serving mistakes, exploded turns, packing, truncation |
| 5. Which tokens receive loss? | `05_loss_mask.py` | the mask proved three ways, tokens scored, normalisation, assistant-only against full-sequence against instruction modelling |
| 6. What does SFT actually teach? | `06_what_sft_teaches.py` | format adherence against N and against few-shot prompting; known against unknown answers |
| 7. The instruction-data taxonomy | `07_data_taxonomy.py` | category lengths, exact checkers, capped mixing |
| 8. How the data gets built | `07_data_taxonomy.py`, `08_data_construction.py` | contamination, near-duplicates, rejection sampling and distinct paths |
| 9. Reasoning SFT | `09_reasoning_sft.py`, `08_data_construction.py` | serial depth, per-step reliability, the outcome filter |
| 10. How much SFT? | `10_how_much_sft.py` | learning rate, epochs, truncation, verbose data, a format-only "drop", template mismatch |
| 11. Where SFT sits in the pipeline | not run | see "What was not measured" |
| 12. What is still unsettled | `05_loss_mask.py` (masking and normalisation), `06_what_sft_teaches.py` (how much data) | two of the six open questions get a small-scale measurement |

## Where the measurements and the post disagree

Each item says where the lab agrees, qualifies or contradicts the post, and which file holds the numbers.

- **A base model "answers with another question rather than Paris" (section 1).** For SmolLM2-135M on a bare question the
  top next token is a newline (0.695) and the model then answers correctly in 83% of cases (12 factual questions). What it
  lacks is a stop and a turn boundary: it repeats the question and answer until the budget ends (0 of 12 stop). Inside a
  chat template it never saw it echoes the question and emits junk. One small model, string-matched. `results/01_the_gap.txt`.
- **Role markers are tokens "the tokenizer never produces from user text" (section 4).** With the default tokenizer call
  the typed string `<|im_end|>` becomes the special token (3 in the sequence, against 2 for an honest message). It holds only
  if the pipeline tokenises user content with special-token parsing off or strips the strings. `results/04_anatomy.txt`.
- **A template mismatch makes the model run past its answer (section 4).** On our own fine-tuned model this is true for
  some mistakes and not others: a missing system turn drops the stop rate from 0.69 to 0.01 and `User:`/`Assistant:` text to
  0.00; a missing newline after `assistant` leaves it at 0.74 yet changes the start of every reply, and a missing newline
  after the user turn leaves it at 0.60. On the released Instruct model no mistake made the stop rate collapse (0.43 to 0.69
  against 0.52 for the correct template), though every one raised the loss on a reference reply by up to 0.5 nats per token and changed the start of
  54% to 100% of replies. `results/10_how_much_sft.txt` (F), `results/04_anatomy.txt` (C).
- **Compounding error grows like eps T^2 (section 3).** The worst-case chain gives slope 1.95 by construction. The learned
  ring task gives local slopes of 1.0 to 1.9 over five seeds, with one seed that never leaves the ring and others that leave
  within 25 steps, so it neither confirms nor contradicts the exponent. Recovery data (noisy execution) and DAgger remove the
  cost; DAgger was not monotone in the iteration. `results/03_behaviour_cloning.txt`.
- **Sampling reproduces the mixture (section 2, "mode-covering").** The 80/20 style mixture comes back as 0.67 on average
  when sampled (range 0.61 to 0.76), so the minority style is over-represented relative to the data, not matched; greedy gives
  0.99. `results/02_same_loss.txt`.
- **Instruction modelling helps when instructions are long, replies short and examples few (section 5, Shi et al.).** Not
  reproduced here. In the long-instruction, short-reply Dolly pool, N = 100 and N = 800, assistant-only was better than
  instruction modelling (w = 0.3) on held-out reply NLL (1.334 against 1.382; 1.174 against 1.216) and on F1 (0.261 against
  0.237 at N = 100; 0.372 against 0.359 at N = 800). Full-sequence loss was clearly worse (F1 0.206 and 0.232). 135M parameters,
  one learning rate, one weight, three seeds, a different recipe from the paper: this does not test the paper. `results/05_loss_mask.txt`.
- **Per-token and per-example normalisation give different models (sections 5 and 12).** Agreed, but the effect is small: on
  Alpaca the per-example loss gave token NLL 1.615 against 1.592 and example NLL 1.665 against 1.681; on the Dolly pool 1.192
  against 1.174 and 1.317 against 1.340. Each normalisation is better on the metric it optimises. The longest 4 of 16 Dolly
  examples carry 61% of the weight under per-token normalisation. Differences are within about one seed standard deviation.
  `results/05_loss_mask.txt`.
- **"Fixed system prompt scored every time" is wasteful (section 5).** True of the token count (the default prompt is about
  10% of an example) but the full-sequence loss was not worse than assistant-only on Alpaca (1.605 against 1.592, within seed
  noise); it hurt on the Dolly pool. `results/05_loss_mask.txt`.
- **Few-shot prompting of the base model approaches SFT (section 6).** Not at this scale. Ten demonstrations gave 0.18 format
  adherence against 0.43 for SFT on 16 examples and 0.74 on 1024. The prompt formats differ (plain text against chat template)
  and the model is 135M. `results/06_what_sft_teaches.txt`.
- **Format alignment generalises "across topics" (section 6).** Adherence to trained formats on held-out topics was 0.43 to
  0.74. The one format never trained reached 0.30 to 0.375 and fell as N grew at fixed steps, so what was learned partly
  tracks the trained formats. `results/06_what_sft_teaches.txt`.
- **Verbatim regurgitation shows before validation loss turns (section 10).** Not seen. At 200 examples validation NLL was
  lowest at 6 epochs (1.628) and rose to 2.950 at 30 epochs, while greedy replies on held-out prompts were never near-copies
  of a training reply (0.000 at every checkpoint). Diversity (distinct-2) fell from 0.915 to 0.770 over the same epochs, steadily
  from epoch 1, so it was moving before the validation minimum. The regurgitation probe asks about held-out prompts, which may
  be the wrong test. `results/10_how_much_sft.txt` (B).
- **A post-SFT benchmark drop is often format (section 10).** Supported in direction: log-likelihood accuracy on ARC-Easy
  stayed at 0.583 to 0.580 while the lenient parse of generated text fell from 0.533 to 0.332. The strict parse is 0.000 for
  both models, so only the lenient parse and the log-likelihood carry information. `results/10_how_much_sft.txt` (E).
- **The outcome filter admits invalid reasoning that does not shrink with k (section 9).** Two results. On GSM8K with a 0.5B
  model, answer-correct solutions with a false arithmetic statement exist at about 1% of the kept ones (0.7% to 1.7% across k in the final run, 0% to 5% at k = 1 across runs), with no trend in k, and the parser has errors of its own.
  On the digit-sum task with an exact step checker, the share is about 0.1% of kept samples, because a slip is carried
  forward and the final digit is wrong unless a second slip cancels it. So the false-positive share depends on the task and
  the answer space; it is not a general tax. `results/08_data_construction.txt`, `results/09_reasoning_sft.txt`.
- **A written trace buys serial depth (section 9).** Agreed. Direct answers fall from 1.00 at n = 2 to 0.10 (chance) at
  n = 10; the trace model is exact up to n = 12. Both fail at lengths beyond training (n = 16, 20: the trace model gets 0.00
  by breaking the format), which the post does not claim either way. `results/09_reasoning_sft.txt`.

## What was not measured

- Nothing runs at the post's scale. Every number quoted from a paper (LIMA, Gekhman et al., Tulu 3, FLAN, Deita, AlpaGasus,
  s1, LIMO, STaR, the R1 recipe, Chu et al.) was not reproduced. Models here are 135M, 360M and 0.5B parameters.
- Section 3's other predictions: that long outputs degrade faster than short ones on a real language model, and that
  jailbreaks are searches for states the demonstrations did not reach, were not tested on a language model. The cloning
  toy is not a language model.
- Rejection-sampling fine-tuning as DAgger on a language model, and Yuan et al.'s claim that distinct paths, not the raw
  count, drive the gain: the lab counts distinct paths among kept samples but does not train on them.
- Self-Instruct and Evol-Instruct generation, and the claim that generation collapses without the diversity filter: only the
  near-duplicate rate of an existing Alpaca set is measured. Embedding-level decontamination is not built.
- Human-written against teacher-written against converted data (FLAN-style task diversity) is not compared by training.
- Capability acquisition and the "long chain-of-thought is elicitation or acquisition" question (section 12, item 1); "SFT
  memorises, RL generalises" (item 2); the recursion across generations (item 6); the pipeline loops of section 11. No RL
  stage exists in this lab.
- Gekhman et al. is reproduced in miniature with one seed and 150-question dev sets, on one model; the hallucination
  measure is accuracy on questions the model knew, not a direct hallucination rate.
- The "model plays both parts" runaway of full-sequence loss, in multi-turn data, was not measured: all training data is
  single-turn.
- Mixture design (section 7) is arithmetic only. The Tulu 3 mixture and Figure 7's saturation curves are not run.
- Catastrophic forgetting is probed only through WikiText NLL and an ARC-Easy log-likelihood on a 135M model.
- Learning-rate and epoch sweeps use one or two seeds and one model; the learning-rate sweep is single-seed per row.
- The strict ARC letter parse is 0.000 for both models and carries no information.
- The loss-mask equivalences (section 5, and multi-turn gradients in section 4) are exact checks of the arithmetic of the
  post, not experiments. Packing was checked with a hand-built 4-D mask on a RoPE model; position resets matter only for
  absolute-position models, which were not used here.
- Single-run items: every learning-rate row, the base-model rows, script 06 part B, scripts 08 and 09 (one seed). The
  three-seed items are scripts 02 (part E), 05 (part D) and 06 (part A); 03 uses five seeds; script 10 parts B to F use two.
