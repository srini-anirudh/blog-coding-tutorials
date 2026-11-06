# C11: Add Memory and Context Engineering to the Agent

The agent from C10 forgets everything when a run ends. This folder gives it memory that survives a restart, and it
fits what the model sees into a token budget. `agentmem` plugs into `harness.Agent` from `../c10-agent-harness`
through two extension points, and the agent loop does not change.

Part of [blog-coding-tutorials](https://github.com/srini-anirudh/blog-coding-tutorials).

## What you build

| Spec item | Module | What it does |
|---|---|---|
| Working memory | `agentmem/working.py` | `TaskState`: goal, current step, plan, open questions |
| Episodic memory | `agentmem/episodic.py`, `records.py` | Append-only `(s, a, o, r, t)` log |
| Semantic memory | `agentmem/semantic.py` | Facts keyed by (subject, attribute), with an edit history |
| Procedural memory | `agentmem/procedural.py` | Reusable rules with a track record |
| Memory writer | `agentmem/writer.py` | `should_store(event)`: novelty, stability, importance, confidence, scope |
| Vector retrieval | `agentmem/embed.py`, `index_naive.py`, `index_fast.py` | Hashing and MiniLM embedders, a reference index and a matrix index |
| Hybrid retrieval | `agentmem/retrieval.py` | `S = alpha S_semantic + beta S_recency + gamma S_importance` |
| Update and supersede | `agentmem/semantic.py` | ADD, UPDATE, SUPERSEDE, DELETE, plus NOOP for a weak contradiction |
| Consolidation | `agentmem/consolidate.py` | Many episodes become one rule |
| Context budget | `agentmem/budget.py` | Packs instructions, task state, memories, events and observations under B |
| Context compression | `agentmem/compress.py`, `tokens.py` | Summarization, extraction, dropping, priority ordering |
| Memory evaluation | `agentmem/bench/`, `scripts/eval_memory.py` | No memory, full transcript, retrieved and oracle |
| Harness integration | `agentmem/integration.py` | `attach_memory(agent, memory, budget)` |

`agentmem/memory.py` holds `AgentMemory`, the one object that ties the write, manage and read steps together.

## Using it on the harness

```python
from agentmem import AgentMemory
from agentmem.integration import attach_memory   # puts ../c10-agent-harness on sys.path
from harness import Agent, ToolRegistry

agent = Agent(model, ToolRegistry([deploy]), workspace)
memory = AgentMemory.load("memory.json")         # or AgentMemory() on the first run
attach_memory(agent, memory, budget=768)
agent.run("Deploy atlas to its usual region.")
memory.end_session()                             # consolidate; the caller decides when a session ends
memory.save("memory.json")
```

`EventLog.subscribe` carries every event to the memory hook, which is the write path. `Agent(context=...)` builds every
prompt, which is the read path.

Pass a `clock` to `attach_memory` in real use. Its default returns a constant, so recency has no effect until you do.

### What this folder needs from C10

`agentmem` imports only top-level `harness` names, all listed under "Public contract" in the C10 README. `call` is the
one exception: it is exported by `harness` but not on that list, and only the tests and `scripts/demo_harness.py` use it.

It also relies on some behaviour that no list of names covers:

- the `Agent(context=...)` argument and the `builder`, `log`, `state`, `tools` and `run` attributes
- `EventLog.subscribe`, `of_type` and `events`
- the event types `UserMessage`, `ToolCall`, `ToolResult` and `Final`, with the opening message at step 0
- `ContextBuilder.instructions` and `sections`, and `AgentState.task`, `plan`, `step` and `notes`

`tests/test_integration.py` pins all of this, so a change on the C10 side fails with a clear message. The folder also
works without C10 next to it: the integration tests skip and everything else still passes. `scripts/eval_memory.py` then
skips its last table, the one that runs through the harness.

## Run the tests

From the repo root:

```bash
python -m pytest c11-agent-memory -m "not slow and not gpu and not network"   # about 15 s
python -m pytest c11-agent-memory -m "network and not slow"                   # downloads MiniLM
python -m pytest c11-agent-memory -m "slow"                                   # real model, two projects, about 2 min
```

## Run everything

Run these from the repo root. All of them run on CPU, and nothing in this folder needs a GPU.

| Command | Time | What it shows | Saved to |
|---|---|---|---|
| `python c11-agent-memory/scripts/run_real_agent.py` | 9 min | Real model, real Agent, memory across two processes | `results/real_agent.*` |
| `python c11-agent-memory/scripts/run_real_agent.py --embedder minilm --conditions retrieved` | 4 min | The same run with sentence embeddings | `results/real_agent_minilm.*` |
| `python c11-agent-memory/scripts/eval_memory.py` | 2 min | Four conditions, budget sweep, ablations, step limit, harness loop | `results/eval_memory.*` |
| `python c11-agent-memory/scripts/eval_lm_reader.py --seeds 3 --device cpu` | 37 min | Same fact contexts, read by a real model | `results/lm_reader.*` |
| `python c11-agent-memory/scripts/demo_harness.py` | seconds | Memory inside the C10 Agent, scripted model | `results/demo_harness.txt` |
| `python c11-agent-memory/scripts/demo_pipeline.py` | seconds | Writer, fact history, index, retrieval, consolidation, packer | `results/demo_pipeline.txt` |
| `python c11-agent-memory/scripts/demo_compress.py` | seconds | Compression strategies side by side | `results/demo_compress.txt` |
| `python c11-agent-memory/scripts/break_it.py` | seconds | Five intentional failures and their symptoms | `results/break_it.txt` |
| `python c11-agent-memory/scripts/compare_trusted.py` | seconds | Index and packer against trusted implementations | `results/trusted.txt` |
| `python c11-agent-memory/scripts/export_figures.py` | seconds | The numbers behind the post's figures | `results/figures.json` |
| `python c11-agent-memory/benchmarks/bench_index.py` | 1 min | Index latency, embedding cost, token estimate | `results/bench_index.*` |

The text files are the stdout of each script. `run_real_agent.py` writes its own.

## Results

All numbers are from an AMD EPYC 7R13 with 8 threads, on a machine shared with other jobs, so treat timings as rough.
The decision tables are deterministic. They match the earlier Apple M5 run exactly, apart from one token in the mean
prompt size of one row.

### The real run

This is the end-to-end proof that `agentmem` works on `harness`. Qwen2.5-0.5B-Instruct (494M parameters, CPU, greedy)
drives `harness.Agent` with a real `deploy` tool that writes to a file. The teaching run and the deploy run are separate
processes, and the store passes between them only as a JSON file. A task counts as correct only if the first deploy in
that file has the right project and region. There are 16 tasks per condition, and 8 of the projects moved region.

Budget B=768 with the hashing embedder:

| Condition | Correct | Moved projects correct | Used the stale value | No deploy | Mean prompt tokens |
|---|---|---|---|---|---|
| no_memory | 0/16 | 0/8 | 0 | 16 | 268 |
| full_transcript | 3/16 | 2/8 | 1 | 0 | 944 |
| retrieved | 13/16 | 8/8 | 0 | 0 | 350 |
| oracle | 16/16 | 8/8 | 0 | 0 | 283 |

Swapping in MiniLM for the embedder gives retrieved 16/16, with the right fact at rank 1 in all 16 prompts.

How to read it:

- **Memory explains the gap.** The model, tools and task are the same in every condition, and only the memory text differs.
- **Retrieval found the fact every time.** The right fact was in all 16 prompts. All three failures have it below the
  top of the list, and the model copied the region from the top line. A fourth task with the fact at rank 6 still passed.
- **Why the rank was low.** The hashing embedder matches on shared words, and every fact here shares "deploy" and
  "region" with the query. Only the project name separates them, so a recently moved project can outscore the right
  fact on recency. Sentence embeddings separate them, which the MiniLM run shows.
- **The transcript hurts this model.** The full transcript holds every fact, and the model still gets 3/16 at 2.7 times
  the prompt size. Only one of the 13 misses used the stale value. The others used a different wrong region.

Limits of this result:

- It is one small model, greedy, with one prompt wording and 16 tasks. Treat the counts as a demonstration, not an
  estimate of accuracy in general.
- `no_memory` scores 0/16 because the model made no tool call at all, not because it guessed wrong. The instructions
  tell it never to guess a region.
- The retrieved store grows during the condition, so later tasks also see earlier deploys. The other conditions do not
  have this.
- The prompt was changed once after a 4-project trial. That trial gave oracle 0/4, because oracle used a different
  header and the model replied with "1". All conditions now share one header, and the instruction telling the model
  what to do when only told a fact was removed. The final numbers come only from the changed prompt.

Each saved file names the harness it ran against, with a hash of its source. `results/real_agent_events.jsonl` holds
the full event record for two projects in each condition.

### Fact and deploy worlds, rule-based reader

The rule reader and rule policy are deterministic, so these tables isolate what the context contains. Full tables are
in `results/eval_memory.txt`.

| Facts, B=1024 | Accuracy | Tokens |
|---|---|---|
| no_memory | 0.578 | 517 |
| full_transcript | 0.749 | 970 |
| retrieved | 0.999 | 582 |
| oracle | 1.000 | 80 |

| Deploy, B=1024 | Success | Steps | Steps on new projects | Tokens |
|---|---|---|---|---|
| no_memory | 0.713 | 5.55 | 6.13 | 137 |
| full_transcript | 0.725 | 5.03 | 5.71 | 937 |
| retrieved | 0.929 | 2.92 | 4.40 | 350 |
| oracle | 1.000 | 1.93 | 1.90 | 102 |

The same policy through the real `harness.Agent` loop gives retrieved 0.923 success in 3.01 steps at B=1024, and it
matches the standalone loop on the no_memory and oracle conditions.

Read the deploy success column with care, because it depends on the step limit. With 8 steps the two conditions differ
(0.713 and 0.929). With 9 steps both reach 1.000. The steps column is the better measure of what memory saves.

Consolidation does not help the default reader: with and without rules it gets 0.929 and 2.92 steps, because that reader
already generalises from other projects' episodes. It does help a literal reader, which goes from 0.881 and 3.59 steps
to 0.904 and 3.30 steps.

### Real model reading the fact contexts

These are the same contexts as the facts table, with only the reader swapped for Qwen2.5-0.5B: 3 seeds and 120 probes
per condition at B=1024. `full_transcript_unbounded` is the whole transcript with no budget.

| Condition | Accuracy | Tokens |
|---|---|---|
| no_memory | 0.442 | 517 |
| full_transcript | 0.542 | 968 |
| full_transcript_unbounded | 0.550 | 2303 |
| retrieved | 0.592 | 582 |
| oracle | 0.875 | 80 |

This is a weaker result than the rule reader's 0.999. Retrieved beats the unbounded transcript by only 0.042 here, with
no confidence interval, so do not read much into that gap on its own. The large gap is between retrieved and oracle.
The model does best when it is given one line, and a retrieved context still gives it several lines to choose from.
Even with the right line alone it reaches only 0.875, so this model sets a limit that memory cannot remove.

### Against trusted implementations

From `results/trusted.txt`:

- `TensorIndex.search` matches `F.cosine_similarity` plus `topk` on 50 of 50 queries (N=2000, D=384, k=10), with a
  largest score difference of 5.96e-08.
- `ContextBudgeter.pack` against an exact brute-force knapsack on 300 random sets of 12 blocks: total value ratio mean
  0.982, worst 0.788. Greedy found the optimum on 179 of 300 sets. It is close on average, and it is not optimal.

### Index benchmark

From `results/bench_index.txt`, D=384, top-10:

| N | Reference index (ms) | Matrix index (ms) | Matrix size (MB) |
|---|---|---|---|
| 1,000 | 16.5 | 0.10 | 1.5 |
| 10,000 | 170.4 | 0.91 | 15.4 |
| 100,000 | skipped | 13.33 | 153.6 |
| 1,000,000 | skipped | 154.79 | 1536.0 |

Embedding 256 short memories takes 30.0 ms with `HashingEmbedder` and 372.9 ms with all-MiniLM-L6-v2. On a small
paraphrase test the cosine with the answering memory is 0.158 and 0.507, against 0.000 and 0.035 for an unrelated one.
Packing 1,000 candidate blocks under B=800 takes 1.67 ms.

## Not measured

- The memory writer's weights are hand-set, not learned. The tests pin its decisions on a fixed set of events.
- The rule-based `RuleFactExtractor` only understands a few sentence patterns. `LLMFactExtractor` exists and has no
  real-model measurement here.
- The real run covers facts only. Procedural memory is tested through the real loop with a scripted model and a rule
  policy, not with the real model.
- `OpenAICompatModel` is tested against a local stub only, because no API keys are used here.

## Dependencies

The repo's shared requirements, plus `../c10-agent-harness` for the integration. The models download from the Hugging
Face Hub on first use. If your default cache is not writable, set `HF_HOME` to a folder you own.
