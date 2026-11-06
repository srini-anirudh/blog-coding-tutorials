# Lab: How Language Models Learn to Use Tools

Code companion for the post [How Language Models Learn to Use Tools](https://srini-anirudh.github.io/blogs/how-language-models-learn-to-use-tools/).

The post argues that a tool call is just tokens, and that what a model learns is to read a tool description, not to
remember a tool. This lab makes that argument small enough to run. It builds six pieces:

1. **The fold.** The agent loop as `x_{k+1} = x_k + y_k + e(Exec(Parse(y_k)))`. All state lives in the context string.
2. **Lookup or interpretation.** A tiny transformer is trained on N distinct synthetic tools and tested on tools it never
   saw. Only N changes. The total number of training examples stays fixed.
3. **Why single-call accuracy lies.** `p^K`, the requirement `0.9^(1/K)`, and an absorbing Markov chain with recovery.
4. **Toolformer's filter.** The five-token loss differential, with the `min` over "no call" and "call without
   result", on GPT-2 and SmolLM2.
5. **Constrained decoding.** A schema mask on the tiny model, plus an exact toy that shows why local masking is not
   the model's distribution conditioned on validity.
6. **Executable verification.** APIGen's format, execution and semantic gates, and the same checks used as a reward.

The tiny model is the Llama-style `TinyLLM` from [C4](../../c04-llm), imported as is. The full production loop
(registry, sandbox, permissions, compaction) is in [C10](../../c10-agent-harness). Training a policy on an
executable reward is in [C12](../../c12-agent-rl). This lab does not rebuild those.

## Layout

| Path | What it holds |
|---|---|
| `toollab/world.py` | The synthetic world: tools, requests, calls, `parse_call`, `execute`, `encode_result` |
| `toollab/loop.py` | The fold over contexts |
| `toollab/model.py` | Build, train and decode the C4 decoder; the `-inf` logit mask |
| `toollab/evaluate.py` | Scores each factor: call or not, which tool, which arguments |
| `toollab/compound.py` | `p^K`, `target^(1/K)`, the on-track / error / dead chain |
| `toollab/toolformer.py` | Weights `w_t`, `L_i(z)`, `L+`, `L-`, a small corpus of call sites |
| `toollab/grammar.py` | Schema mask for the call syntax; exact local-versus-global comparison with `alpha` |
| `toollab/verify.py` | The three gates and the reward |
| `scripts/01..06` | One script per experiment, listed below |
| `results/` | Real outputs. Each file starts with its hardware and command |

## Setup

From the repository root, after `pip install -e ".[dev]"`. Checkpoints go to `runs/`, which git ignores.
Script 05 downloads GPT-2 and SmolLM2 from the Hugging Face Hub the first time.

## Run everything

```bash
L=labs/how-language-models-learn-to-use-tools
python -m pytest $L -m "not slow and not gpu and not network"   # 15 tests, about 20 s
python -m pytest $L -m network                                  # 1 test, downloads GPT-2

python $L/scripts/02_compounding.py         > $L/results/02_compounding.txt
python $L/scripts/05_toolformer_filter.py --models openai-community/gpt2 HuggingFaceTB/SmolLM2-135M \
    HuggingFaceTB/SmolLM2-360M              > $L/results/05_toolformer_filter.txt
python $L/scripts/03_tool_diversity.py      # trains N = 16, 128, 1024, 4000; writes results/tool_diversity_N*.json
python $L/scripts/03_tool_diversity.py --table > $L/results/03_tool_diversity.txt
python $L/scripts/01_fold.py                > $L/results/01_fold.txt
python $L/scripts/04_constrained_decoding.py > $L/results/04_constrained_decoding.txt
python $L/scripts/06_verify_and_reward.py   > $L/results/06_verify_and_reward.txt
```

Script 03 is the slow one. Each N trains for 6,000 steps of batch 32. On a shared 4-thread slice of an AMD EPYC 7R13
that took 12 to 15 minutes per N. The four values of N were run as four parallel processes. Pass `--n-tools 1024` to run one,
or `--steps 2000` for a quick, undertrained pass. Scripts 01, 04 and 06 read the checkpoints that 03 writes.

## Which script covers which section of the post

| Post section | Script |
|---|---|
| The shape of one turn; A tool call is only tokens | `01_fold.py` |
| Does the tool live in the weights or the context?; The training distribution picks the winner | `03_tool_diversity.py` |
| Decomposing the decision | `03_tool_diversity.py` (decide / select / args columns) |
| Now multiply; The escape hatch is recovery, not precision | `02_compounding.py` |
| Toolformer: the criterion, the weights, the min | `05_toolformer_filter.py` |
| A hundred thousand tools (selection gets harder with more tools in context) | `03_tool_diversity.py` (by tools in context) |
| Generalising to tools you never trained on | `03_tool_diversity.py` (unseen split) |
| Manufacturing the trajectories: the APIGen pattern | `06_verify_and_reward.py` |
| Syntax is free, semantics is not | `04_constrained_decoding.py` |
| Rewards you can execute | `06_verify_and_reward.py` |

## The synthetic world

A tool has a random two-syllable name, one purpose word and one to three typed fields. A request names a purpose and
carries values. The right output is a call to the tool in context with that purpose, each value bound to the field of
its type. If no tool in context has the purpose, the right output is `NONE`.

```
<tools> <tool> ke tu : play ( se su : item ) <tool> fe lo : track ( ki fi : person , fu de : count )
<user> for play you coffee <call>
target: ke tu ( se su = coffee ) <eos>
```

Names and field names of the unseen tools come from a pool that never appears in training. So a model can call an
unseen tool only by copying it from the context. The loss is ordinary next-token cross-entropy on the call span only;
the prompt is masked with `-100`, as the post describes.

## Results

Every number below was measured on CPU: an AMD EPYC 7R13 on a shared machine with a high load. Script 03 used 4 threads
per process. The other scripts used 8. torch 2.15.0, transformers 5.22.2. Raw output is in `results/`.

### Lookup or interpretation (`03_tool_diversity.py`)

The tiny model has 3 layers, width 64 and 0.17M parameters. Each run sees M = 192,000 examples (6,000 steps of 32).
Every run has one to three tools in context. "Unseen" means tools whose names and field names never appeared in training.
Each split has 300 requests. About a fifth of the requests have no matching tool, so `NONE` is the right answer.

| N tools | M/N | exact, seen tools | exact, unseen tools | exact on unseen *calls* | unseen: right tool, given a call | unseen: right args, given the tool | final train loss | train time |
|---|---|---|---|---|---|---|---|---|
| 16 | 12,000 | 0.983 | 0.167 | 0.000 | 0.000 | - | 0.002 | 717 s |
| 128 | 1,500 | 0.993 | 0.180 | 0.000 | 0.000 | - | 0.001 | 710 s |
| 1,024 | 188 | 0.327 | 0.377 | 0.350 | 0.744 | 0.548 | 0.70 | 719 s |
| 4,000 | 48 | 0.290 | 0.253 | 0.203 | 0.678 | 0.403 | 0.94 | 892 s |

How to read it. With few tools (large M/N), the model is nearly perfect on its own tools and calls no unseen tool
correctly. Its 0.17 on unseen tools comes only from the `NONE` requests. With many tools (small M/N), the model is far
weaker overall, but its seen and unseen scores are about the same. That is the signature of reading the description
rather than recalling a row.

What the memorising models do on an unseen tool:
- They mostly answer `NONE`. On unseen requests that needed a call, the N = 16 model says `NONE` 96% of the time and
  the N = 128 model 85% of the time. From inside the context, a tool it has no row for looks like a tool that does not exist.
- When they do call, the name is not in the context. For N = 16, 91% of wrong names are names of its own training
  tools; for N = 128, all of them are. That is the post's "hallucinate the nearest neighbour".
- For N = 4,000 the "training tool" share (0.63) means little. Its training tools use most of the two-syllable names,
  so a random wrong name is likely to be one.

The interpreting models are not finished training. Their loss was still falling at 6,000 steps, and more diverse data
learned more slowly: N = 4,000 is behind N = 1,024 at the same step count. The table supports "diversity removes the
seen/unseen gap". It does not show that diversity helps accuracy on the training tools within a fixed budget. Here it
hurts.

Selection gets harder with more tools in context (exact on unseen calls, N = 1,024): 0.549 with one tool, 0.344 with
two, 0.135 with three. For N = 4,000: 0.402, 0.156, 0.041.

### Compounding and recovery (`02_compounding.py`)

| Quantity | Value |
|---|---|
| 0.95^10 | 0.5987 |
| 0.95^50 | 0.0769 |
| p needed for 0.9 task success at K = 10 / 20 / 50 | 98.9519% / 99.4746% / 99.7895% |
| K = 50, p = 0.95, no recovery | 0.0769 |
| K = 50, one more point of precision, p = 0.96 | 0.1299 |
| K = 50, p = 0.95, half of all errors recovered (q = 0.5) | 0.2837 |
| K = 50, p = 0.95, q = 0.8 | 0.5928 |

These are exact (matrix powers). A 100,000-run Monte Carlo agrees to within 0.004 in every row. The chain is
on track to error with 1 - p, error back to on track with q, error to dead with 1 - q. Success means being on track
after K steps. The post's K = 20 value of 99.48% rounds to 99.47%.

### Toolformer's filter (`05_toolformer_filter.py`)

32 call sites (12 QA, 12 Calculator, 8 Calendar), each with three candidate calls. "Answer in call" means the call's own
text already contains the answer, for example `QA("Is Nairobi the capital of Kenya?") -> Yes`. Losses are the weighted
loss over the next five tokens, in nats. The weights are 1, 0.8, 0.6, 0.4, 0.2, normalised.

Calls kept at tau_f = 1.0, with the `min` and without it (`L(empty) - L+` alone):

| Model | Candidate | QA kept, with min / without | Calculator | Calendar |
|---|---|---|---|---|
| GPT-2, 124M | useful | 4/12 / 12/12 | 12/12 / 12/12 | 8/8 / 8/8 |
| GPT-2, 124M | answer in call | 0/12 / 12/12 | 0/12 / 12/12 | 0/8 / 7/8 |
| GPT-2, 124M | irrelevant | 0/12 / 9/12 | 0/12 / 0/12 | 0/8 / 0/8 |
| SmolLM2, 135M | useful | 0/12 / 3/12 | 0/12 / 11/12 | 8/8 / 8/8 |
| SmolLM2, 135M | answer in call | 0/12 / 3/12 | 0/12 / 0/12 | 0/8 / 4/8 |
| SmolLM2, 360M | useful | 0/12 / 0/12 | 0/12 / 1/12 | 8/8 / 8/8 |
| SmolLM2, 360M | answer in call | 0/12 / 1/12 | 0/12 / 0/12 | 0/8 / 4/8 |

What it shows:
- Without the `min`, 31 of 32 "answer in call" candidates pass for GPT-2 at tau_f = 1.0, and 9 of 12 irrelevant QA
  calls pass too. Those calls help only by being a good prompt. With the `min`, none of them pass.
- A call pays rent only where the model does not already know the answer. GPT-2's loss on the capital is 3.95 nats
  without help, and some useful QA calls pass. SmolLM2-360M's is 0.44 nats, and none pass. The Calendar call passes
  for every model, because no model can know today's date.
- The threshold matters. At tau_f = 0.5 SmolLM2-135M keeps all 12 useful Calculator calls. At 1.0 it keeps none.
  All thresholds are in `results/05_toolformer_filter.txt`.

This is the filter only. Nothing is fine-tuned on the kept calls. The post's yield table and the 775M emergence result
were not reproduced.

### Constrained decoding (`04_constrained_decoding.py`)

The exact toy: the model prefers tool A (0.6), but A's likely value has the wrong type. The valid strings are `Ay` and
`Bx`.

| String | p(y given y is valid) | local masking | local masking weighted by alpha |
|---|---|---|---|
| Ay | 0.1364 | 0.6000 | 0.1364 |
| Bx | 0.8636 | 0.4000 | 0.8636 |

KL(exact to local) is 0.463 nats. Greedy decoding with the mask returns `Ay`, a valid, confident, wrong string.
alpha(A) = 0.10 and alpha(B) = 0.95. Weighting each step by alpha recovers the exact conditional.

The tiny models on 300 unseen-tool requests. "Schema" forces tool names, field names and punctuation from the tools in
context and allows any value. "Typed" also restricts each value to its field's type.

| N tools | decoding | exact | exact on calls | right tool, given a call | right args, given the tool |
|---|---|---|---|---|---|
| 16 | none / schema / typed | 0.210 / 0.217 / 0.217 | 0.000 / 0.000 / 0.000 | - | - |
| 1,024 | none / schema / typed | 0.343 / 0.423 / 0.600 | 0.289 / 0.387 / 0.613 | 0.691 / 0.768 / 0.768 | 0.475 / 0.572 / 0.906 |
| 4,000 | none / schema / typed | 0.260 / 0.327 / 0.450 | 0.175 / 0.260 / 0.417 | 0.668 / 0.922 / 0.922 | 0.344 / 0.367 / 0.590 |

N = 128 is in the results file. It behaves like N = 16.

What it shows:
- The mask cannot make the memorising model call anything. `NONE` is legal, so it keeps answering `NONE`, and its
  exact rate on calls stays at 0.000. Whether to call is not a syntactic constraint.
- For the interpreting models, many errors were copy errors in a field name. The mask removes them. The typed mask
  raises N = 1,024 from 0.289 to 0.613 exact on calls.
- What remains is semantic. Even with the typed mask, N = 1,024 picks the wrong tool on 23% of calls, and 9% of
  its calls with the right tool still carry a wrong value of the right type.
- Unmasked parse rates are already 0.99 or higher here, so this small grammar shows no case where masking made
  outputs worse. The post's "can trade correctness for parseability" claim was shown only on the exact toy.

### Verification gates and reward (`06_verify_and_reward.py`)

100 unseen-tool requests, 8 samples each at temperature 1. Each sample goes through the gates in order; the table
gives the share that fails at each gate. r = r_fmt + r_exec + r_match, each 0 or 1.

| N tools | fail format | fail execution | fail semantic | kept | mean r | groups of 8 with any reward spread |
|---|---|---|---|---|---|---|
| 16 | 0.024 | 0.000 | 0.886 | 0.090 | 2.04 | 4/100 |
| 128 | 0.170 | 0.000 | 0.740 | 0.090 | 1.75 | 14/100 |
| 1,024 | 0.491 | 0.189 | 0.130 | 0.190 | 1.02 | 70/100 |
| 4,000 | 0.774 | 0.020 | 0.129 | 0.077 | 0.51 | 63/100 |

What it shows:
- The cheap gates do most of the work for the interpreting models: 68% of N = 1,024 samples die at format or execution.
  Only 13% reach the semantic gate and fail there.
- The memorising model passes the cheap gates and fails the expensive one. Its `NONE` answers are well formed and
  run, but they are wrong.
- This reward has a flaw, and the numbers expose it. `NONE` always earns r_fmt + r_exec = 2. So the model that cannot
  call unseen tools scores a higher mean reward (2.04) than the one that can (1.02). Reward design decides what is
  learned, which is ToolRL's point. A fixed version would give `NONE` no format or execution credit when a call was
  needed.
- A group-relative update needs reward spread inside a group. Only 4 of 100 groups give the memorising model any
  signal, against 70 of 100 for N = 1,024.

The post's third reward term is r_outcome. In this world each request has one correct call and no side effects, so
outcome and match are the same event. The lab scores "it ran" as the middle term instead.

### The fold (`01_fold.py`)

A scripted model makes a typo, reads the error that the harness wrote back, and retries. The final context holds
every step, including the mistake. Part 2 lets the N = 16 and N = 1,024 models emit calls on six fixed unseen-tool
requests with three tools in context. The harness parses each span and runs it. The N = 16 model answers `NONE` to all
six. The N = 1,024 model emits five calls and one `NONE`, and all six are wrong in a name or a field. That matches its 0.135 exact rate with
three tools in context. The harness turns each one into an error message.

### Tests

15 offline tests pass in about 20 s, and the 1 `network` test passes in about 50 s (AMD EPYC 7R13). There are no
`slow` or `gpu` tests.

## What was not measured

- No result in this lab reproduces a paper's numbers. The tool counts in the post (APIBench, ToolBench, APIGen, ToolACE),
  Toolformer's yield table and its 775M emergence result, ToolLLM's 66.7% vs 47.0%, APIGen's leaderboard results and
  ToolRL's gains are all at a scale this lab does not reach.
- The tiny model is trained once per N with one seed. Differences of a few points between runs are within noise.
- The diversity runs are not trained to convergence. High-N models were still improving at 6,000 steps (see the loss
  curves in the JSON files). The claim tested is the gap between seen and unseen tools, not the absolute accuracy.
- Tool retrieval (recall@k as a ceiling) is not built. The lab only measures how selection changes with one to three
  tools in context.
- The semantic gate in script 06 is a rule that compares the call with the ground truth. APIGen uses a model as judge.
- No policy is trained on the reward. Script 06 only measures the reward and how often a group of samples would give a
  group-relative update any signal. C12 trains with such advantages.
- The `alpha` correction is computed exactly on a toy distribution only. It is intractable for a real model, which is
  the post's point.
- No GPU was used. Every number is from CPU.
