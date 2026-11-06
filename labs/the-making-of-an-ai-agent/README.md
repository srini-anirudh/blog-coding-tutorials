# Lab: The Making of an AI Agent

This lab is the code companion to the post
[The Making of an AI Agent](https://srini-anirudh.github.io/blogs/the-making-of-an-ai-agent/). The post argues that a
loop over a world with state changes what post-training is. This lab builds the post's central claims as small
experiments you can run:

1. **Mask the world's tokens.** We train the same tiny model on agent trajectories, with and without the loss mask.
   About 94% of each trajectory was written by the world, not the model.
2. **Compounding error.** In a corridor world, behaviour cloning's cost grows with the square of the horizon.
   DAgger's grows linearly, until mistakes become irreversible.
3. **Credit assignment.** GRPO gives every action in a trajectory the same advantage. GiGPO's anchor-state groups
   give each action its own. We compare them on a chain task where states repeat.
4. **Reliability.** pass^k against mean success, computed from the policies trained in step 3.
5. **Four kinds of state.** A context compaction erases the record of a write, and the agent writes again. Each
   harness fix is counted.
6. **Tool retrieval is RAG over actions.** We count the context cost of a tool catalogue, and show how retrieval
   recall caps task success.
7. **Plans have a half-life.** We compare plan-then-act, ReAct and a hybrid in a folder that another writer edits.

Everything is small and synthetic. The papers' benchmark numbers (FireAct, AgentTuning, ARTIST, GiGPO on ALFWorld,
SWE-smith, τ-bench) are **not** reproduced here. The lab shows each mechanism small, so you can see it happen.

Other parts of the series build the full systems. [C10](../../c10-agent-harness) builds the agent harness.
[C11](../../c11-agent-memory) builds memory and context compaction. [C12](../../c12-agent-rl) trains an agent with
behaviour cloning, GRPO and a critic in a toy filesystem. This lab does not rebuild those. It covers what the post adds.

## Layout

```
labs/the-making-of-an-ai-agent/
├── agentcraft/
│   ├── common.py          # hardware line, results files, Markdown tables
│   ├── masking.py         # synthetic trajectories, a 2-layer causal transformer, masked vs unmasked SFT
│   ├── compounding.py     # the corridor, the Ross-Bagnell sum, behaviour cloning and DAgger
│   ├── credit.py          # chain task with repeated states, GRPO and GiGPO advantages, exact success by DP
│   ├── reliability.py     # pass@k and pass^k, exact and unbiased estimators
│   ├── state_layers.py    # calendar, auth token, context compaction, write log, idempotent tool
│   ├── tool_retrieval.py  # synthetic tool catalogue, JSON schemas, TF-IDF retriever
│   └── planning.py        # shared folder with another writer, plan-then-act, ReAct, hybrid
├── scripts/               # 01_ ... 07_, one per experiment
├── results/               # what the scripts wrote; each file starts with its hardware and command
└── tests/
```

## Setup

From the repository root:

```bash
pip install -e ".[dev]"
```

Only `torch` and `numpy` are needed, plus `transformers` for one tokenizer. `06_tool_retrieval.py` counts schema
tokens with the SmolLM2-135M tokenizer if it is in the local Hugging Face cache. If it is not, the script counts regex
word pieces instead and says so in its output.

## Run everything

```bash
# Tests (about 15 s on CPU). One more test downloads a tokenizer: add -m network to run it.
python -m pytest labs/the-making-of-an-ai-agent -m "not slow and not gpu and not network"

python labs/the-making-of-an-ai-agent/scripts/01_masking.py          # 6 runs x 3000 steps; about 35 min on 8 cores
python labs/the-making-of-an-ai-agent/scripts/02_compounding.py      # 5 s
python labs/the-making-of-an-ai-agent/scripts/03_credit.py           # 20 runs, about 1 min on 8 cores
python labs/the-making-of-an-ai-agent/scripts/04_reliability.py      # reads results/03_credit.json
python labs/the-making-of-an-ai-agent/scripts/05_state_layers.py     # 1 s
python labs/the-making-of-an-ai-agent/scripts/06_tool_retrieval.py   # 30 s
python labs/the-making-of-an-ai-agent/scripts/07_planning.py         # 12 s
```

`01_masking.py --steps 600 --seeds 1` gives a quick look. By step 600 the masked model has begun to learn the copy
and the unmasked one has not. No GPU is used: the script has a `--device cuda` path that was not run.

## Which script covers which section of the post

| Post section | Script | What it shows |
|---|---|---|
| Trajectories are the training data: "You must mask the world's tokens" | `01_masking.py` | The unmasked loss fits the world and learns the policy more slowly. |
| The gradient, with the environment removed | `01_masking.py` | The gradient of the world's tokens is many times larger than the policy's. |
| Imitation teaches paths: the compounding-error argument | `02_compounding.py` | BC cost against the Ross-Bagnell sum; DAgger is linear. |
| Why agents have it worse than the textbook case | `02_compounding.py` | With irreversible mistakes, DAgger is quadratic again. |
| Credit assignment over long horizons: the four levers | `03_credit.py` | GRPO vs GiGPO, and what happens when states cannot be matched. |
| Post-training becomes an environment problem: pass^k | `04_reliability.py` | The same mean success can give very different pass^8. |
| Four kinds of state | `05_state_layers.py` | Compaction causes duplicate writes; three fixes; an expiring token. |
| Tool retrieval is RAG over actions | `06_tool_retrieval.py` | Context cost s·\|T\|; all-three-retrieved as the success ceiling. |
| Planning versus execution | `07_planning.py` | Plan success (1-q)^L; ReAct pays 2x the calls; the hybrid gets both. |

## Results

Every number names its hardware. Raw output is in `results/`, under the name of the script that wrote it. Each file
starts with a `# hardware:` line and the command.

Tests on AMD EPYC 7R13, 8 threads, torch 2.15.0: 21 passed offline, 1 `network` test passed (it loads the
SmolLM2-135M tokenizer). There are no `slow` or `gpu` tests.

### 1. Masking the world's tokens (`results/01_masking.txt`)

AMD EPYC 7R13, CPU only, 6 runs in parallel at 2 threads each. A 2-layer, 64-wide causal transformer is trained from
scratch for 3,000 steps (batch 32, AdamW) on synthetic episodes: the task names a key, the model writes `search K`,
the world returns 110 tokens (a Markov-chain passage with 4 key-value facts written into it), and the model must
write `answer K V` with the value that followed K. The model wrote 5.7% of the scored tokens. Three seeds, mean ± std,
on 512 held-out episodes. Chance on the value is 1/16 = 0.06.

| Objective | Answer accuracy | NLL on model-written tokens | NLL on world-written tokens |
|---|---|---|---|
| Masked: loss on the model's tokens only | 0.308 ± 0.013 | 0.246 ± 0.022 | 10.73 ± 1.05 |
| Unmasked: loss on every token | 0.094 ± 0.039 | 0.393 ± 0.004 | 1.654 ± 0.002 |

Both models were given the same data and the same steps. Masked training learns the copy within 750 steps and then
plateaus near 0.30. The unmasked model has not got there by step 3,000 (0.09). It has instead fit the world: its NLL on
the world's tokens is 1.65, against 10.7 for the masked model, which never trained on them. The loss the unmasked
model minimises is mostly a model of the environment, which is the post's point that including observation tokens
"changes the objective".

The gradient split (unmasked runs, measured on one batch every 250 steps) shows why. The gradient from the world's
tokens is 13 to 16 times the norm of the policy's gradient from step 500 on (12.7 at its lowest, at step 1750). The cosine between the full gradient
and the policy part is about 0.13. Gradient clipping at 1.0 then scales the policy's share down further.

Two limits. Neither model reaches high accuracy in 3,000 steps: the best masked seed is about 0.32, so this
does not show the end state, only the speed and the objective. And this is a toy where the world is learnable
structure. A real observation is longer and noisier, so the effect here may be smaller or larger than at scale.
The paper-level claim (Search-R1: models trained with the retrieved-token mask outperform those without) is not
reproduced.

### 2. Compounding error (`results/02_compounding.txt`)

AMD EPYC 7R13, 8 threads, CPU. Per-step error ε = 0.02. 20,000 episodes per cell. Cost is the expected number of
steps that end off the expert's path.

| H | Ross-Bagnell sum | ε H(H+1)/2 | BC, measured | DAgger, measured | DAgger, irreversible mistakes | ε H |
|---|---|---|---|---|---|---|
| 3 | 0.118 | 0.120 | 0.121 | 0.061 | 0.122 | 0.060 |
| 10 | 1.037 | 1.100 | 1.034 | 0.207 | 1.040 | 0.200 |
| 50 | 18.84 | 25.50 | 18.98 | 1.014 | 18.82 | 1.000 |
| 100 | 57.50 | 101.0 | 58.15 | 2.038 | 57.39 | 2.000 |

The log-log slope of cost against H, between H = 5 and H = 50, is 1.83 for BC and 0.99 for DAgger. With irreversible
mistakes, DAgger's slope is 1.80. At ε = 0.02, BC spends 4% of its steps off the path at H = 3 and 38% at H = 50.

Two things to notice. First, ε H(H+1)/2 is an upper bound that holds only while εH is small. By H = 100 it is almost
twice the exact sum, because by then many runs have already failed. Second, DAgger fixes the problem only because it
can step back to the path. Once a mistake cannot be undone, labelled recovery states do not help. Then the only fix
is to make fewer mistakes. This is the post's point about agents that delete rows or send emails.

### 3. Credit assignment: GRPO vs GiGPO (`results/03_credit.txt`)

AMD EPYC 7R13, 8 threads, CPU. 64 tasks, 4 stages, 4 actions per stage (1 advances, 1 is a dead end, 2 are no-ops),
horizon 12, reward 1 only on reaching the end. Tabular softmax policy, 16 tasks x 8 rollouts per iteration. Five
seeds, mean ± std. Success is the exact probability, computed by dynamic programming over the policy.

| Method | Success, iter 25 | Success, iter 50 | Success, iter 200 | No-op probability, iter 100 | No-op probability, iter 200 |
|---|---|---|---|---|---|
| GRPO (episode advantage only) | 0.367 ± 0.023 | 0.775 ± 0.014 | 0.986 ± 0.001 | 0.182 ± 0.004 | 0.127 ± 0.005 |
| GiGPO, ω = 1 | 0.397 ± 0.022 | 0.823 ± 0.012 | 0.994 ± 0.000 | 0.032 ± 0.001 | 0.007 ± 0.000 |
| GiGPO, half of states unmatched | 0.356 ± 0.012 | 0.798 ± 0.004 | 0.993 ± 0.000 | 0.056 ± 0.004 | 0.014 ± 0.001 |
| GiGPO, no state ever matched | 0.361 ± 0.025 | 0.770 ± 0.023 | 0.986 ± 0.000 | 0.182 ± 0.003 | 0.124 ± 0.004 |

The step-level signal mostly removes wasted actions. In a successful trajectory, GRPO credits every no-op with the
same positive advantage as the actions that did the work. So at iteration 200 the GRPO policy still puts 13% of its
probability on no-ops, against 0.7% for GiGPO. The gain in success is smaller: 0.823 against 0.775 at iteration 50.
When the harness cannot recognise repeated states, GiGPO's anchor groups have one member and it becomes GRPO. That is
the limitation the post names, for environments whose state is a rendered web page.

The script also prints one group of four rollouts with both advantages written per step. It shows a nuance. Within an
anchor group, GiGPO compares discounted returns. An action that advanced, in a rollout that later hit a dead end,
still gets a negative A_S. It ranks actions by what followed them, not by what they caused.

### 4. Reliability: pass^k (`results/04_reliability.txt`)

AMD EPYC 7R13, 8 threads, CPU. Exact values from the per-task success probabilities of seed 0 in step 3.

| Agent | Mean success | pass^4 | pass^8 | pass@8 |
|---|---|---|---|---|
| GRPO, iteration 50 | 0.773 | 0.436 | 0.226 | 0.978 |
| GiGPO, iteration 50 | 0.816 | 0.530 | 0.325 | 0.984 |
| 0.65 on every task | 0.650 | 0.179 | 0.032 | 1.000 |
| 0.65 of tasks always solved, the rest never | 0.650 | 0.650 | 0.650 | 0.650 |

The last two rows have the same mean success and very different pass^8. pass^k measures whether success is
consistent within a task. An agent with 77% mean success passes all eight trials on only 23% of tasks. Simulated
trials with the unbiased estimator C(c, k)/C(n, k), using n = 16, give pass^8 within 0.016 of the exact values.

### 5. Four kinds of state (`results/05_state_layers.txt`)

AMD EPYC 7R13, 8 threads, CPU. 50 episodes, each asking for 6 calendar events with 1 to 3 email reads after each.
The context holds 12 lines. Compaction keeps the task, a summary of the emails read, and the last 6 lines. The policy
is scripted, not a language model: it treats an event as done only if the context shows a confirmation for it.

| Harness | Duplicate events per episode | Tool calls | Finished |
|---|---|---|---|
| Naive: the context is the only record | 5.10 | 22.96 | 100% |
| + write log kept outside the context, re-injected after compaction | 0 | 17.86 | 100% |
| + policy lists events before each write | 0 | 30.74 | 100% |
| + the create tool refuses duplicates | 0 | 22.96 | 100% |
| No compaction | 0 | 17.86 | 100% |
| Token expires after 15 calls, no refresh | 0 | 181.46 | 10% |
| Token expires after 15 calls, the harness refreshes it | 0 | 18.76 | 100% |

The summariser keeps what looks like content and drops what looks like bookkeeping. The write confirmations are
bookkeeping, so after a compaction 5 of the 6 events are created a second time. Each fix lives in a different layer.
The write log is persistent memory. Listing events reads the environment, at 72% more calls than the write log. Idempotency is tool
state. With an expiring token and no refresh, the agent retries the failing call until the 200-step limit. Fixing
that is harness work, not model work.

### 6. Tool retrieval (`results/06_tool_retrieval.txt`)

AMD EPYC 7R13, 8 threads, CPU. A synthetic catalogue of (domain, verb, object) tools with JSON schemas. The mean
schema is 106.3 tokens with the SmolLM2-135M tokenizer. Putting every schema in the prompt costs 1.06 M tokens at
10,000 tools, and 1.75 M at ToolBench's 16,464. The top 8 cost 850.

Each task needs three tools, and the request uses a synonym for each word half the time. The retriever is TF-IDF
cosine over the whole request. "All three retrieved" is the ceiling on task success.

| Catalogue | k | Per-tool recall | All three retrieved (ceiling) | recall^3 |
|---|---|---|---|---|
| 1,000 | 3 | 0.570 | 0.086 | 0.185 |
| 1,000 | 8 | 0.817 | 0.548 | 0.545 |
| 1,000 | 16 | 0.943 | 0.856 | 0.839 |
| 10,000 | 8 | 0.410 | 0.052 | 0.069 |
| 10,000 | 32 | 0.723 | 0.346 | 0.378 |
| 10,000 | 64 | 0.815 | 0.538 | 0.541 |

A per-tool recall of 0.82 leaves only 55% of tasks solvable. With a 10x larger catalogue and the same k, the ceiling
falls from 0.548 to 0.052. Splitting the request into its three clauses and retrieving per clause did not help this
retriever (see the results file). In this catalogue each request names its domain, and a domain holds about
|T|/50 tools. So at 1,000 tools, k = 32 already covers the whole domain and recall is 1.0. That is a property of
this synthetic catalogue, not a general result.

### 7. Planning versus execution (`results/07_planning.txt`)

AMD EPYC 7R13, 8 threads, CPU. Another writer moves an unread file with probability q = 0.05 after every tool call.
20,000 episodes per row.

| Files L | (1-q)^L | Plan-then-act success | Calls | ReAct success | Calls | Hybrid success | Calls |
|---|---|---|---|---|---|---|---|
| 2 | 0.902 | 0.900 | 4.0 | 1.000 | 6.1 | 1.000 | 4.2 |
| 8 | 0.663 | 0.664 | 10.0 | 1.000 | 18.3 | 1.000 | 10.8 |
| 32 | 0.194 | 0.194 | 34.0 | 1.000 | 66.4 | 1.000 | 36.4 |

Plan-then-act succeeds with probability (1-q)^L, so a plan has a half-life of ln 0.5 / ln(1-q) = 13.5 steps here.
ReAct always succeeds and nearly doubles the calls. The hybrid re-plans only when a read fails. It always succeeds,
at 7% more calls than the plan. That works because every disruption here is loud: a moved file gives a read error.
A silent change, such as a value edited after the agent read it, would defeat the hybrid. That case is not built here.

## What was not measured

- None of the papers' numbers are reproduced: FireAct, AgentTuning, ARTIST, ToolRL, Search-R1, GiGPO on ALFWorld and
  WebShop, WebArena, SWE-smith, τ-bench, ToolSandbox. Every environment here is synthetic and small.
- No language model drives an agent here. The policies are a 2-layer transformer (step 1), lookup tables (steps 2
  and 3) and scripted rules (steps 5 and 7). [C12](../../c12-agent-rl) trains a transformer policy in a tool
  environment with behaviour cloning and GRPO.
- The process-reward and learned-critic levers of the post are not built here. C12 has a critic with GAE.
- Synthetic environments, distillation from a teacher, and the data-mixture coefficients λ are not measured.
- Tool retrieval uses TF-IDF, not a trained neural retriever like ToolLLM's, and a templated catalogue, not ToolBench.
