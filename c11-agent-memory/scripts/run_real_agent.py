"""The real run: a real model, the real C10 Agent, and memory that survives a restart.

    python c11-agent-memory/scripts/run_real_agent.py [--projects 16] [--budget 768] [--embedder minilm]

Nothing here is scripted. A local Hugging Face model drives `harness.Agent`, with a real
tool that writes to disk, and the outcome is read back from that file.

    process 1  TEACH   the user states each project's deploy region, then moves half of
                       them. The memory hook writes through the harness event log.
                       The store is saved to disk and the process exits.
    process 2  DEPLOY  a new process loads the store. For each project a fresh Agent is
                       told "Deploy <project> to its usual region." under four conditions.

Only the context differs between conditions, so a difference in outcome is caused by it.
Writes results/real_agent.txt, results/real_agent.json and results/real_agent_events.jsonl.
With --embedder minilm the names start real_agent_minilm instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from agentmem import AgentMemory, TransformerEmbedder  # noqa: E402
from agentmem.bench.hardware import hardware_label, use_threads  # noqa: E402
from agentmem.integration import attach_memory  # noqa: E402  (puts ../c10-agent-harness on sys.path)

import harness  # noqa: E402
from harness import Agent, AgentConfig, ContextBuilder, HFLocalModel, ModelResponse, ToolRegistry, tool  # noqa: E402

INSTRUCTIONS = ("You are a deploy agent. To deploy, call the deploy tool once with the project and its region. "
                "Take the region from your memory, and never guess one. "
                "After the deploy tool has returned, reply with one short sentence and call no tool.")
NAMES = ["atlas", "cobalt", "dune", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kestrel", "lumen",
         "meridian", "nimbus", "onyx", "pollux", "quartz"]
REGIONS = ["eu-west-1", "us-east-2", "ap-south-1", "sa-east-1", "ca-central-1", "me-south-1", "eu-north-1", "af-south-1"]
CONDITIONS = ("no_memory", "full_transcript", "retrieved", "oracle")
HEADER = "\n\n## Relevant memory\n"  # the header the retrieved condition renders, so only the content differs


def world(n: int) -> list[dict[str, str | None]]:
    """Every even-numbered project moves region in session 2, so its first value goes stale."""
    out = []
    for i, name in enumerate(NAMES[:n]):
        first = REGIONS[i % len(REGIONS)]
        moved = REGIONS[(i + 3) % len(REGIONS)] if i % 2 == 0 else None
        out.append({"project": name, "first": first, "moved": moved, "truth": moved or first})
    return out


def harness_fingerprint() -> tuple[str, str]:
    """Which harness actually ran: its folder relative to this one, and a hash of its source."""
    root = Path(harness.__file__).resolve().parent
    digest = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        digest.update(f.relative_to(root).as_posix().encode() + f.read_bytes())
    return f"../{root.parent.name}/{root.name}", digest.hexdigest()[:16]


class Recording:
    """Wraps any model client: keeps the first prompt of a run and the exact token counts."""

    def __init__(self, inner) -> None:
        self.inner, self.name = inner, inner.name
        self.reset()

    def reset(self) -> None:
        self.system, self.calls = None, []

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        self.system = self.system or messages[0]["content"]
        response = self.inner.complete(messages, tools)
        self.calls.append({"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens,
                           "latency_s": round(self.inner.last_latency_s, 2)})
        return response


def deploy_tools(ledger: Path) -> ToolRegistry:
    @tool
    def deploy(project: str, region: str) -> str:
        """Deploy a project to a region.

        Args:
            project: The project name.
            region: The cloud region, such as eu-west-1.
        """
        with ledger.open("a") as f:
            f.write(json.dumps({"project": project, "region": region}) + "\n")
        return f"deployed {project} to {region}"
    return ToolRegistry([deploy])


def load_model(args: argparse.Namespace) -> Recording:
    import transformers

    transformers.utils.logging.disable_progress_bar()
    use_threads(8)
    return Recording(HFLocalModel(args.model, device="cpu", max_new_tokens=args.max_new_tokens))


def make_embedder(args: argparse.Namespace):
    return TransformerEmbedder() if args.embedder == "minilm" else None  # None: the hashing default


def fact_rank(memory_section: str, project: str) -> int | None:
    """Where the right fact landed in the prompt's memory list, 1 = top."""
    lines = memory_section.splitlines()
    return next((i + 1 for i, ln in enumerate(lines) if ln.startswith(f"- {project} deploy region:")), None)


def teach(args: argparse.Namespace, state: Path) -> None:
    """Process 1. Everything the store learns arrives through the harness event log."""
    model = load_model(args)
    memory, transcript, runs = AgentMemory(make_embedder(args)), [], []
    ledger = state / "teach_ledger.jsonl"
    tools = deploy_tools(ledger)
    statements = [(1, f"The deploy region for {p['project']} is {p['first']}.") for p in world(args.projects)]
    statements += [(2, f"Heads up: we moved the deploy region for {p['project']} to {p['moved']}.")
                   for p in world(args.projects) if p["moved"]]
    count = 0
    session = 1
    for number, text in statements:
        if number != session:
            memory.end_session()
            session = number
        count += 1
        clock = lambda: number + count / 1000  # noqa: B023
        agent = Agent(model, tools, state, context=ContextBuilder(instructions=INSTRUCTIONS),
                      config=AgentConfig(max_steps=2, checkpoint_on_start=False))
        attach_memory(agent, memory, args.budget, session=lambda: number, clock=clock)  # noqa: B023
        result = agent.run(text)
        replies = [e.data["text"] for e in agent.log.of_type("ModelResponse")]
        calls = [c for e in agent.log.of_type("ModelResponse") for c in e.data["tool_calls"]]
        transcript += [f"user: {text}"] + [f"assistant: {r}" for r in replies if r]
        runs.append({"statement": text, "status": result.status, "reply": replies[-1][:120] if replies else "",
                     "tool_calls": [c["arguments"] for c in calls]})
    memory.end_session()
    memory.save(state / "memory.json")
    (state / "transcript.txt").write_text("\n".join(transcript))
    fingerprint = harness_fingerprint()
    (state / "teach.json").write_text(json.dumps({"pid": os.getpid(), "harness": fingerprint, "runs": runs}))


def deploy_phase(args: argparse.Namespace, state: Path) -> None:
    """Process 2. A new interpreter: all it has of process 1 is the files in `state`."""
    model = load_model(args)
    projects = world(args.projects)
    transcript = (state / "transcript.txt").read_text()
    rows, events = [], []
    embedder = make_embedder(args)
    for condition in args.conditions:
        memory = AgentMemory.load(state / "memory.json", embedder) if condition == "retrieved" else None
        for n, p in enumerate(projects):
            project, truth = p["project"], p["truth"]
            ledger = state / f"{condition}_{project}.jsonl"
            instructions = INSTRUCTIONS
            if condition == "full_transcript":
                instructions += HEADER + transcript
            elif condition == "oracle":
                instructions += f"{HEADER}- {project} deploy region: {truth}"
            agent = Agent(model, deploy_tools(ledger), state, context=ContextBuilder(instructions=instructions),
                          config=AgentConfig(max_steps=3, checkpoint_on_start=False))
            if memory is not None:
                attach_memory(agent, memory, args.budget, session=lambda: 3,
                              clock=lambda: 3 + (len(memory.episodes) + 1) / 1000)  # noqa: B023
            model.reset()
            task = f"Deploy {project} to its usual region."
            result = agent.run(task)
            done = [json.loads(ln) for ln in ledger.read_text().splitlines()] if ledger.exists() else []
            first = done[0] if done else None
            rows.append({
                "condition": condition, "project": project, "truth": truth, "stale_value": p["first"] if p["moved"] else None,
                "fact_in_prompt": f"{project} deploy region: {truth}" in model.system or (
                    condition == "full_transcript" and truth in transcript),
                "first_deploy": first, "deploys": len(done),
                "correct": bool(first and first["project"] == project and first["region"] == truth),
                "stale": bool(first and p["moved"] and first["region"] == p["first"]),
                "status": result.status, "steps": result.steps, "final_text": result.final_text[:160],
                "memory_section": model.system.split("## Relevant memory\n")[-1].split("\n\n")[0] if "## Relevant memory" in model.system else "",
                "estimated_context_tokens": agent.log.of_type("ModelResponse")[0].data["context"],
                "model_calls": model.calls,
            })
            print(f"{condition:<16}{project:<10} truth={truth:<13} deployed={first['region'] if first else '-':<14}"
                  f"{'ok' if rows[-1]['correct'] else 'WRONG'}", file=sys.stderr, flush=True)
            if n < 2:  # the full event record of two projects per condition is enough to read
                events += [{"condition": condition, "project": project, **asdict(e)} for e in agent.log.events]
    (state / "deploy.json").write_text(json.dumps({"pid": os.getpid(), "harness": harness_fingerprint(), "rows": rows,
                                                   "hardware": hardware_label(), "model_params": sum(
        q.numel() for q in model.inner.model.parameters())}))
    (state / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))


def summarise(rows: list[dict], conditions: list[str]) -> dict[str, dict]:
    out = {}
    for cond in conditions:
        rs = [r for r in rows if r["condition"] == cond]
        calls = [c for r in rs for c in r["model_calls"]]
        moved = [r for r in rs if r["stale_value"]]
        out[cond] = {"tasks": len(rs), "correct": sum(r["correct"] for r in rs), "moved": len(moved),
                     "correct_on_moved": sum(r["correct"] for r in moved), "stale": sum(r["stale"] for r in rs),
                     "no_deploy": sum(not r["first_deploy"] for r in rs), "fact_in_prompt": sum(r["fact_in_prompt"] for r in rs),
                     "first_prompt_tokens": sum(r["model_calls"][0]["prompt_tokens"] for r in rs) / max(1, len(rs)),
                     "s_per_call": sum(c["latency_s"] for c in calls) / max(1, len(calls)), "calls": len(calls)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--projects", type=int, default=16)
    ap.add_argument("--budget", type=int, default=768)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--embedder", choices=["hashing", "minilm"], default="hashing")
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--phase", choices=["teach", "deploy"])
    ap.add_argument("--state", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    if args.phase:
        (teach if args.phase == "teach" else deploy_phase)(args, args.state)
        return

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp)
        for phase in ("teach", "deploy"):
            subprocess.run([sys.executable, __file__, "--phase", phase, "--state", str(state), "--model", args.model,
                            "--projects", str(args.projects), "--budget", str(args.budget),
                            "--max-new-tokens", str(args.max_new_tokens), "--embedder", args.embedder,
                            "--conditions", *args.conditions], check=True)
        first, second = json.loads((state / "teach.json").read_text()), json.loads((state / "deploy.json").read_text())
        memory_state = json.loads((state / "memory.json").read_text())
        events = (state / "events.jsonl").read_text()
    rows, table = second["rows"], summarise(second["rows"], args.conditions)
    (location, hash_start), (_, hash_end) = first["harness"], second["harness"]
    lines = [f"model {args.model} ({second['model_params'] / 1e6:.0f}M parameters), {second['hardware']}, budget B={args.budget}, embedder {args.embedder}",
             f"harness {location}, source hash {hash_end}, unchanged during the run: {hash_start == hash_end}",
             f"teach and deploy ran in separate processes: {first['pid'] != second['pid']}",
             f"store after teaching: {sum(f['status'] == 'active' for f in memory_state['facts'])} active facts, "
             f"{sum(f['status'] == 'superseded' for f in memory_state['facts'])} superseded, "
             f"{len(memory_state['episodes'])} tool steps from the teaching runs", "",
             f"{'condition':<18}{'correct':>9}{'moved ok':>10}{'stale':>7}{'no deploy':>11}{'fact in prompt':>16}{'prompt tokens':>15}{'s/call':>8}"]
    for cond, r in table.items():
        lines.append(f"{cond:<18}{r['correct']:>5}/{r['tasks']:<3}{r['correct_on_moved']:>6}/{r['moved']:<3}{r['stale']:>7}{r['no_deploy']:>11}"
                     f"{r['fact_in_prompt']:>13}/{r['tasks']:<2}{r['first_prompt_tokens']:>15.0f}{r['s_per_call']:>8.1f}")
    lines += ["", "one line per task: condition, project, what it should deploy to, what landed in the ledger, rank of the right fact in a retrieved prompt"]
    for r in rows:
        got = r["first_deploy"]
        lines.append(f"{r['condition']:<16}{r['project']:<10}{r['truth']:<14}{got['project'] + ' ' + got['region'] if got else '(no deploy)':<30}"
                     f"{'ok' if r['correct'] else 'stale' if r['stale'] else 'wrong':<7}"
                     + (f"rank {fact_rank(r['memory_section'], r['project'])}" if r["condition"] == "retrieved" else ""))
    lines.append(f"\nwall time {time.perf_counter() - t0:.0f}s")
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(exist_ok=True)
    stem = "real_agent" if args.embedder == "hashing" else f"real_agent_{args.embedder}"
    (args.out / f"{stem}.txt").write_text(text + "\n")
    (args.out / f"{stem}.json").write_text(json.dumps(
        {"model": args.model, "embedder": args.embedder, "hardware": second["hardware"], "budget": args.budget, "harness": location,
         "harness_hash": hash_end, "harness_unchanged": hash_start == hash_end, "separate_processes": first["pid"] != second["pid"],
         "summary": table, "teach_runs": first["runs"], "rows": rows}, indent=1))
    (args.out / f"{stem}_events.jsonl").write_text(events)


if __name__ == "__main__":
    main()
