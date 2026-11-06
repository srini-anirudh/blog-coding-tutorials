# C10: Build an Agent Harness From Scratch

Code for the tutorial [Build an Agent Harness From Scratch](https://srini-anirudh.github.io/blogs/build-an-agent-harness-from-scratch/).
Repo: https://github.com/srini-anirudh/blog-coding-tutorials/tree/main/c10-agent-harness

We start from an 8-line loop and add each missing production piece until a model that can call
tools becomes a coding agent: a tool registry, structured calls, file and shell tools, a patch
applier that fails loudly, verification tools, sandboxes, permissions, a context builder,
repository search and a map, a plan, a JSONL event log, and compaction.

The package is `harness`. It uses only the standard library. PyTorch and transformers are needed
only for `HFLocalModel`, which imports them when you use it.

## Layout

| Path | What it holds |
|---|---|
| `harness/loop_minimal.py` | The 8-line loop |
| `harness/tools.py`, `parsing.py`, `messages.py` | `@tool`, `ToolRegistry`, schema generation, call parsing and validation |
| `harness/fs_tools.py`, `search.py`, `repomap.py` | Read, list and write files, ripgrep, git grep or Python search, symbol index, repository map |
| `harness/editing.py` | Whole file, search/replace and unified diff, with an applier that refuses instead of corrupting |
| `harness/shell.py`, `sandbox.py` | The shell tool, `LocalSandbox`, `DockerSandbox` |
| `harness/verify.py` | `run_tests`, `run_linter`, `run_typecheck`, `build`, with one result shape |
| `harness/permissions.py` | Read, workspace-write, external and dangerous, plus approval |
| `harness/context.py`, `compaction.py`, `truncation.py`, `tokens.py` | What the model sees each turn |
| `harness/planning.py`, `events.py`, `replay.py`, `tracing.py`, `git_tools.py` | Plan, JSONL log, replay, latency table, checkpoints |
| `harness/models/` | `ScriptedModel`, `HFLocalModel`, `OpenAICompatModel` |
| `harness/agent.py`, `toolset.py` | `Agent` and the default toolset |
| `examples/toy_repo/` | A small project with a real bug, the agent's task |

## Public contract

C11 builds on these names, all importable from `harness`: `Agent`, `AgentConfig`, `AgentResult`,
`AgentState`, `ContextBuilder`, `ToolRegistry`, `tool`, `ToolContext`, `ToolOutput`, `ToolError`,
`EventLog`, `Event`, `Message`, `ToolCall`, `ModelResponse`, `ModelClient`, `ScriptedModel`,
`HFLocalModel`, `OpenAICompatModel`, `PermissionPolicy`, `Risk`, `LocalSandbox`, `DockerSandbox`,
`Limits`, `coding_tools`.

## Run the tests

From the repo root:

```bash
python -m pytest c10-agent-harness -m "not slow and not gpu and not network"   # about 30 s
python -m pytest c10-agent-harness -m "network and not slow"                   # downloads a tokenizer
python -m pytest c10-agent-harness -m "slow"                                   # loads Qwen2.5-0.5B on CPU
```

The real-container tests are in the default run. They skip, with the reason in the `-rs` output,
when `docker` is not installed, when the daemon cannot be reached, when `python:3.12-slim` is not
present, or when a plain container cannot start. They never pull, so an offline run stays offline.
Run `docker pull python:3.12-slim` once to enable them. That check uses none of the sandbox's own
flags, so a bug in `DockerSandbox` makes a test fail. It cannot turn into a skip.

## Run the scripts and benchmarks

Redirect a command's output to the `.txt` file to reproduce it. Event logs and JSON files are written by the script.

| Command | Result file |
|---|---|
| `python c10-agent-harness/scripts/minimal_loop.py` | `results/minimal_loop.txt` |
| `python c10-agent-harness/scripts/show_schemas.py` | `results/schemas.txt` |
| `python c10-agent-harness/scripts/demo_fix_test.py` | `results/demo_output.txt`, `results/demo_events.jsonl` |
| `python c10-agent-harness/scripts/demo_docker_sandbox.py` | `results/docker_sandbox.txt` |
| `python c10-agent-harness/scripts/compare_trusted.py` | `results/compare_trusted.txt` |
| `python c10-agent-harness/scripts/run_hf_agent.py --device cpu` | `results/hf_output.txt`, `results/hf_events.jsonl` |
| `python c10-agent-harness/scripts/run_openai_agent.py` (set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `HARNESS_MODEL`) | `results/openai_stub_output.txt` |
| `python c10-agent-harness/benchmarks/bench_truncation.py` | `results/bench_truncation.txt` |
| `python c10-agent-harness/benchmarks/bench_edit_formats.py` | `results/bench_edit_formats.txt` |
| `python c10-agent-harness/benchmarks/bench_compaction.py` | `results/bench_compaction.txt`, `results/compaction_series.json` |
| `python c10-agent-harness/benchmarks/bench_overheads.py` | `results/bench_overheads.txt` |

`results/openai_stub_output.txt` came from a local stub server that replayed scripted replies. It
tests the client and the loop. It says nothing about any real model.

## The Docker sandbox

`DockerSandbox` runs each command in a fresh container: `--rm`, `--network none`, `--cpus`,
`--memory`, `--pids-limit`, `--cap-drop ALL`, `no-new-privileges`, the workspace mounted at
`/workspace`, and a name that starts with `ct-c10-`.

It runs as the host user (`--user uid:gid`). The first real run taught us why. With
`--cap-drop ALL`, the image's root user has no `CAP_DAC_OVERRIDE`, so the kernel treats it as a
stranger to the host user's files. It got `cat: a.txt: Permission denied` in a mode-700 workspace,
and it could not write to a mode-755 one. The pull had worked. The host uid fixes both, and the
files it writes stay owned by the person who owns the workspace.

`python:3.12-slim` has no pytest, and with the network off nothing can install it. `run_tests`
then reports `unavailable`, not a test failure. Use an image that has your test tools. The demo
gets by without one: it copies pytest's pure-Python packages into the workspace and sets
`PYTHONPATH` through `extra_env`.

## Results

All numbers come from an AMD EPYC 7R13, 8 threads, CPU only, with Docker 29.1.4.

| Measurement | Result | File |
|---|---|---|
| Scripted agent fixes the toy repo | done after 13 steps, 6 of 6 tests pass | `demo_output.txt` |
| Same fix with every `run_tests` in a container | 4 passed and 2 failed, then 6 passed | `docker_sandbox.txt` |
| `true` in a fresh container, median of 10 | 412 ms | `docker_sandbox.txt` |
| `true` in `LocalSandbox`, median of 20 | 5.7 ms, 13.2 ms with network denial | `bench_overheads.txt` |
| 512 MB allocation under a 128 MB limit | killed, exit 137, reported as `memory`, in 0.59 s | `docker_sandbox.txt` |
| Busy loop under a 1 s CPU limit | killed, exit 152, in 2.61 s | `docker_sandbox.txt` |
| `sleep 30` with a 2 s timeout | timed out, container removed, in 2.71 s | `docker_sandbox.txt` |
| Fork loop under `--pids-limit 16` | blocked after 14 forks | `docker_sandbox.txt` |
| Harness overhead per step, scripted model, no-op tool | 1.11 ms | `bench_overheads.txt` |
| Silent corruptions in 1400 edits with simulated slips | whole file 600, search/replace 0, strict diff 0, anchored diff 0 | `bench_edit_formats.txt` |
| Our anchored applier against `git apply`, 400 diffs | same result on all 400 | `compare_trusted.txt` |
| 80 steps, cumulative prompt tokens, no budget and a 2000-token budget | 1150281 and 123219, 89% saved | `bench_compaction.txt` |
| Failing assertion kept in a 20000-line log | head + tail 0%, head + tail + rescue 100% | `bench_truncation.txt` |
| Qwen2.5-0.5B on the toy repo, 8 threads on CPU | did not fix the bug: one bad edit, then it gave up after 2 steps | `hf_output.txt` |

Qwen2.5-0.5B is a tiny model, and this result is the honest one. The harness stopped its edit
with a clear error, and the model did not recover.

Nothing in this folder needs a GPU.

## Not verified

- A real container with `Limits(network=True)`. Only its argv is tested, because the machine rules
  for these runs allow `--network none` only.
- A successful first-time pull through `DockerSandbox`. The image was already present. The failed
  pull path is tested against a closed local port.
- macOS and Docker Desktop. The Docker tests ran on Linux with cgroup v2. The cgroup test skips on
  other hosts.
- Any real remote model. No API key was used.
