# Lab: Inside an AI Agent Harness

Code companion for the post [Inside an AI Agent Harness](https://srini-anirudh.github.io/blogs/inside-an-ai-agent-harness/).
Repo: https://github.com/srini-anirudh/blog-coding-tutorials/tree/main/labs/inside-an-ai-agent-harness

The post argues that most of a coding agent's cost and behaviour is set by the harness, not by the model.
This lab runs seven of its arguments as small programs:

1. The loop re-sends its whole context at every step, so input grows with the square of the step count.
   Prefix caching turns that back into a line, but only if the old prompt is an exact prefix of the new one.
2. The same prefix reuse on a real transformer's KV cache: exact logits, and the time it saves.
3. Code as the action: one program in place of one tool call per file.
4. Edit protocols: a landed edit costs m / p tokens, and a failed one stays in the context.
5. Observation design: which truncation policy keeps the one line that explains the failure.
6. The permission ladder's order, and a model of approval fatigue.
7. Turn length as a geometric variable, caps, and a stop rule that needs the tests to pass.

The C10 tutorial, [Build an Agent Harness From Scratch](../../c10-agent-harness/), builds the harness
itself: tools, the loop, sandboxes, the event log and compaction. This lab does not rebuild those. It
imports C10's edit appliers (`harness.editing`), its token estimate (`harness.tokens`) and its
truncation policy (`harness.truncation`), and spends its code on the post's arithmetic.

## Setup

From the repo root, with the repo's environment (`pip install -e ".[dev]"`):

```bash
python -m pytest labs/inside-an-ai-agent-harness -m "not slow and not gpu and not network"
```

The package is `harness_lab`. Importing it puts `c10-agent-harness` on `sys.path`, so `import harness`
works. Only `kv_reuse.py` needs a model download. It uses `HuggingFaceTB/SmolLM2-135M-Instruct` on
CPU, and `Qwen/Qwen2.5-0.5B-Instruct` for the GPU run. The tests use a tiny random Llama and download
nothing.

## Run everything

```bash
python labs/inside-an-ai-agent-harness/scripts/loop_cost.py                 # about 30 s
python labs/inside-an-ai-agent-harness/scripts/kv_reuse.py                  # about 10 min on 8 CPU threads
python labs/inside-an-ai-agent-harness/scripts/code_action.py               # a few seconds
python labs/inside-an-ai-agent-harness/scripts/edit_cost.py                 # about 2 min
python labs/inside-an-ai-agent-harness/scripts/observation.py               # about 10 s
python labs/inside-an-ai-agent-harness/scripts/permissions_and_stopping.py  # a few seconds
# optional, one GPU:
python labs/inside-an-ai-agent-harness/scripts/kv_reuse.py --device cuda \
    --model Qwen/Qwen2.5-0.5B-Instruct --dtype bfloat16 --prefix 8192 --delta 1024 --steps 23 \
    --out kv_reuse_gpu.txt
```

Each script writes `results/<name>.txt`. The first line of each file names the hardware, the second the command.

## Results

CPU numbers: AMD EPYC 7R13, 8 threads, torch 2.15. GPU numbers: NVIDIA H100 80GB HBM3, torch 2.15, CUDA 13.2.
Token counts do not depend on hardware. Times do.

| Claim in the post | What the lab measured | Hardware | File |
|---|---|---|---|
| A 60-step turn with P0 = 14k and delta = 2k sends about 4.5M input tokens | 4,500,000 sent, equal to the closed form at every K tried | any (token count) | `loop_cost.txt` |
| With full prefix caching it collapses to P0 + K delta | cold cache: 134,000 computed = P0 + K delta. Warm cache: 120,000 | any | `loop_cost.txt` |
| T_eff for a cache hit rate h | simulation within 1 standard error of the closed form at every h tried, 400 seeds | any | `loop_cost.txt` |
| Editing an early block breaks the cache for everything after it | state change at step 20: append computes 1,232 tokens, edit in place 20,688 | any | `loop_cost.txt` |
| Editing item 6 of 40 costs the rest of the list | 35 of 40 items recomputed (items 6 to 40) | any | `loop_cost.txt` |
| A nondeterministic tool order breaks caching | 20 tool names from a Python `set`, 30 hash seeds: 30 distinct orders, 15.3% of tokens cached, against 94.8% in sorted order | any | `loop_cost.txt` |
| Compaction is a scheduled cache miss, and a bad threshold costs more | K = 200: billed input falls from 4.66M (no compaction) to 1.03M at a 30k threshold, then rises to 1.16M, plus 264k summary tokens, at 22k | any | `loop_cost.txt` |
| Tool-context tax K M tau = 1.4M tokens | 1,440,000 schema tokens eager, 14,160 deferred | any | `loop_cost.txt` |
| Prefix reuse is exact | SmolLM2-135M, fp32, 24 requests up to 5,120 tokens: max logit difference 4.0e-05 against a fresh forward pass, same next token in 24 of 24 | AMD EPYC 7R13, 8 threads | `kv_reuse_cpu.txt` |
| Prefix reuse makes the turn cheaper | 16 steps, 1,024-token prefix plus 256 tokens per step: no reuse computes 51,200 tokens in 90.8 s, reuse 5,120 tokens in 16.2 s (5.6x). One prefix token edited at step 9: that step computes 3,264 tokens and takes 8.5 s instead of 1.0 s | AMD EPYC 7R13, 8 threads | `kv_reuse_cpu.txt` |
| The same on a GPU, with a longer context | Qwen2.5-0.5B bf16, 23 steps, 8,192-token prefix plus 1,024 per step, last request 31,744 tokens: no reuse computes 471,040 tokens in 2.51 s, reuse 31,744 in 0.60 s (4.2x). Edit at step 12: that step computes 20,416 tokens and takes 265 ms instead of 25 ms; whole turn 0.84 s (3.0x). Logits identical to a fresh pass in all 35 requests (difference 0.0 in bf16), same next token in 35 of 35 | NVIDIA H100 80GB HBM3 | `kv_reuse_gpu.txt` |
| Code actions need Theta(1) round trips, tool calls Theta(n) | 80 files: 114 model calls and 1.02M input tokens, against 2 calls and 1,045 tokens. Same final repo | any (C10 token estimate) | `code_action.txt` |
| E[m] = m / p per landed edit | measured within 4% of m / p in all 21 size and format cells. Mean tries within 0.05 of 1 / p | AMD EPYC 7R13 (token count) | `edit_cost.txt` |
| Small file: rewrite it. Large file: patch it | at 10 lines a whole-file rewrite (118 tokens) beats a unified diff (169). From 25 lines on, both patch formats win, and at 800 lines the rewrite costs 9,354 | any | `edit_cost.txt` |
| A failed patch costs more than 1/p | unified diff: 169 to 196 output tokens per landed edit, but its failed attempts and error messages cost another 1,962 to 2,375 context tokens over the next 20 steps | any | `edit_cost.txt` |
| Truncation policy is a correctness feature | grep answer at 50% of 40,001 lines: lost by first N, head and tail, summary and C10 rescue. Kept by spill to handle, at 931 tokens | any | `observation.txt` |
| | build error at 50% of 20,002 lines: lost by first N and head and tail, kept by summary (65 tokens) and C10 rescue | any | `observation.txt` |
| Deny and ask rules outrank the permission mode | 4 of 24 action and mode cells change if the mode is checked first. All 4 are a bypass overriding a deny, an ask or a hook | any | `permissions_and_stopping.txt` |
| Mean vigilance falls with the number of prompts | with a0 = 1 and h = 20 (chosen, not measured): 0.90 at 5 prompts, 0.14 at 200. Cutting 200 prompts by 84% raises it to 0.59 | any | `permissions_and_stopping.txt` |
| E[L] goes from 10 to 50 between q = 0.90 and 0.98 | sampled 10.03 and 49.98. A cap of 40 holds q = 0.98 at 27.7 | any | `permissions_and_stopping.txt` |
| "No tool call" is not "done" | toy agent, 10,000 tasks: the naive rule ends 4,328 in false victory, the conjunctive rule 0 | any | `permissions_and_stopping.txt` |

## Post sections and scripts

| Post section | Script | Module |
|---|---|---|
| Why the naive loop is quadratic | `scripts/loop_cost.py` parts 1 and 2, `scripts/kv_reuse.py` | `prefix_cache.py`, `loop_cost.py`, `kv_reuse.py` |
| The append-only invariant | `scripts/loop_cost.py` parts 3 and 4, the `reuse + edit` lane of `scripts/kv_reuse.py` | `loop_cost.py`, `kv_reuse.py` |
| Choosing an action space (code beats JSON) | `scripts/code_action.py` | `code_action.py` |
| Editing is a protocol, not a talent | `scripts/edit_cost.py` | `edit_cost.py` (uses C10 `harness.editing`) |
| Observation design | `scripts/observation.py` | `observation.py` (uses C10 `harness.truncation`) |
| Permissions and the arithmetic of fatigue | `scripts/permissions_and_stopping.py` parts 1 and 2 | `permissions.py` |
| Budgets, stopping, and the illusion of done | `scripts/permissions_and_stopping.py` parts 3 and 4 | `stopping.py` |
| Compaction: the lossy codec in the loop | `scripts/loop_cost.py` part 5 | `loop_cost.py` |
| Progressive disclosure and the tool-context tax | `scripts/loop_cost.py` part 6 | `loop_cost.py` |
| The loop, the event log, sandboxes, verification | see C10 | `c10-agent-harness/harness/` |

## How to read the numbers

Tokens in `loop_cost.py` are fresh integer ids, so two blocks match only when they are the same block.
The cache is block-hashed, like vLLM's automatic prefix caching. Hosted APIs work on the same exact-prefix
rule, but their block sizes and minimum lengths differ.

The compaction sweep counts input tokens and the summary's output tokens separately. It does not price
them against each other. The "no compaction" row assumes a context window of over 400k tokens, which is
a reference line, not a real option.

In `edit_cost.py` the token costs are measured on real files (C10's own source). The success
probabilities p are parameters: 0.98 for a whole file, 0.85 for search and replace, 0.7 for a unified
diff. The post calls its own values schematic. These are too. Each failure is real, though: the C10
applier, or a guard, refuses the slipped edit and its error message is what enters the context.

In `observation.py` all three outputs come from real processes: `grep -rn` over generated files, a
`pytest -s` run, and a printed build log. "Summary" is a parser per tool. For grep it reports counts per
file, which is why it loses a single match.

`permissions.py` is a small model of the ladder the Claude Agent SDK documents, not the SDK. The
vigilance curve's shape is the post's model. Its parameters are chosen here.

`stopping.py`'s termination test uses a toy agent with a hidden state. The verifier is a function of that
state. It shows the mechanism. Its 4,328 is not comparable with the post's 14 of 100, which came from
real harnesses and models.

## Not measured

- The post's production figures: the 93% approval rate, the 84% prompt reduction, the 0.4% and 17%
  auto-approval rates, the 24 of 25 exfiltrations. They come from telemetry and an internal exercise.
- The harness-effect study (40x token spread across Goose, OpenCode and OpenHands-SDK) and the
  Terminal-Bench 2.0 comparison. Both need real harnesses, real models and the benchmark.
- SWE-agent's interface ablation, CodeAct's 20.7 points, Aider's 20% to 61%, and the 150k to 2k MCP
  example. The lab shows the mechanisms behind them on scripted agents, not the results.
- Any real model choosing actions. Every agent here is scripted or simulated, so the lab says nothing
  about how often a model slips.
- Hosted-API prefix caching. The lab's cache is local. It follows the same exact-prefix rule.
- Compaction quality: what a summary loses. The post says this has no clean public measurement. The lab
  counts only its token cost.
