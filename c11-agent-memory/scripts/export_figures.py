"""Export the numbers behind the post's interactive figures from the real code.

    python c11-agent-memory/scripts/export_figures.py

results/figures.json holds: the budget packer run at every B on a grid, the three hybrid
retrieval terms of every memory, the index tensors for one query, and one real deploy
trace. The page only displays these numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from agentmem import AgentMemory, Block, BudgetError, ContextBudgeter, MemoryEvent, MemoryItem  # noqa: E402
from agentmem import count_tokens  # noqa: E402
from agentmem.bench.deploy_eval import DeployPolicy, run_deploy  # noqa: E402
from agentmem.bench.deploy_world import make_world  # noqa: E402
from agentmem.records import reset_ids  # noqa: E402

FACTS = [
    (1.0, "The deploy region for atlas is eu-west-1."), (1.1, "The owner for atlas is priya."),
    (1.2, "The database for cobalt is postgres."), (2.0, "Remember: never deploy on Fridays."),
    (3.0, "The test command for atlas is `pytest -x`."), (4.0, "We moved the deploy region for atlas to us-east-2."),
    (5.0, "The release branch for atlas is stable."), (6.0, "The owner for cobalt is chen."),
    (7.0, "Important: the staging cluster is shared, so run smoke tests only."),
]
QUERY = "which region does atlas deploy to"
LOG = "\n".join(f"[{i}/7] {stage} atlas: done in {i * 3}.{i}s (cache hit)" for i, stage in
                enumerate(["resolve", "fetch", "compile", "test", "package", "push", "rollout"], 1)) + \
    "\nAuthError: registry rejected the push token (401)"


def search_tensors(mem: AgentMemory) -> dict:
    """The index matrix restricted to the buckets the query touches: all a dot product can see."""
    idx = mem.retriever.index
    q = mem.embedder.encode([QUERY])[0]                 # [D]
    buckets = q.nonzero().flatten().tolist()
    slots = [idx.slot_of[i] for i in mem.retriever.items]
    rows = idx.matrix[slots][:, buckets]                # [N, len(buckets)]
    return {"dim": idx.dim, "buckets": buckets, "q": [round(float(v), 3) for v in q[buckets]],
            "texts": [it.text for it in mem.retriever.items.values()],
            "rows": [[round(float(v), 3) for v in row] for row in rows],
            "cos": [round(float(v), 3) for v in idx.matrix[slots] @ q], "shape": [len(slots), idx.dim]}


def deploy_trace(seed: int = 0, budget: int = 1024) -> list[dict]:
    """One real run of the deploy world: what the policy saw and did at every step."""
    log: list[dict] = []
    policy = DeployPolicy()

    def recording(ctx: str, task_id: str, project: str) -> str:
        action = policy(ctx, task_id, project)
        memories = [ln[2:] for ln in ctx.split("## Relevant memory\n")[-1].split("\n\n")[0].splitlines()] \
            if "## Relevant memory" in ctx else []
        log.append({"task": task_id, "project": project, "tokens": count_tokens(ctx), "action": action,
                    "rules": sum(m.startswith("rule:") for m in memories), "memories": [m[:96] for m in memories[:3]]})
        return action

    world = make_world(seed)
    results = run_deploy(world, "retrieved", budget, policy=recording)
    return [{**e, "steps": next(r.steps for r in results if r.task.id == e["task"]),
             "quirk": world.quirks[e["project"]]} for e in log]


def main() -> None:
    reset_ids()
    mem = AgentMemory()
    for t, text in FACTS:
        mem.observe(MemoryEvent(text, "user", t, entities=["atlas"] if "atlas" in text else []))
    items, comp = mem.retriever.components(QUERY, now=8.0)
    hybrid = [{"text": it.text, "kind": it.kind, "semantic": round(float(c[0]), 4), "recency": round(float(c[1]), 4),
               "importance": round(float(c[2]), 4)} for it, c in zip(items, comp)]

    # A realistic candidate pool: everything the packer may choose from at one agent step.
    blocks = [Block("instructions", "You are a deploy agent. Tools: deploy, migrate, login, build_assets. "
                    "If a deploy fails, fix the cause and retry.", required=True, compressible=False, id="instructions"),
              Block("task", "goal: deploy atlas\ncurrent step: fix the failed deploy\nopen question: why was the token rejected?",
                    required=True, id="task state"),
              Block("request", "Deploy atlas to its current region.", required=True, compressible=False, id="request")]
    rule = "rule: when `deploy {entity}` fails with AuthError, run `login {entity}` and retry (from 3 episodes over 3 projects)"
    mem.retriever.add(MemoryItem("procedural", rule, 7.5, importance=1.0, id="rule"))
    hits = mem.retrieve("deploy atlas AuthError", 8.0, k=7)
    blocks += [Block("memory", f"- {h.item.text}", value=h.score, order=r, id=h.item.text.split(" (from")[0]) for r, h in enumerate(hits)]
    for i, ev in enumerate(["[s8t1] `deploy atlas` -> AuthError", "[s8t1] `clear_cache atlas` -> ok", "[s8t1] `deploy atlas` -> AuthError"]):
        blocks.append(Block("events", ev, value=0.3 + 0.7 * (i + 1) / 3, order=i, id=f"step {i + 1}: " + ev.split("] ")[1]))
    blocks.append(Block("observation", LOG, value=0.9, id="deploy log"))

    budgeter = ContextBudgeter()
    grid = []
    for budget in range(40, 521, 8):
        try:
            p = budgeter.pack(blocks, budget)
        except BudgetError as e:
            grid.append({"budget": budget, "error": str(e)})
            continue
        grid.append({"budget": budget, "total": p.total, "by_section": p.tokens_by_section,
                     "in": [b.id for b in p.blocks if b.id not in p.compressed], "compressed": p.compressed,
                     "out": [b.id for b in p.dropped]})
    candidates = [{"id": b.id, "section": b.section, "tokens": budgeter.cost(b), "value": round(b.value, 3),
                   "density": round(b.value / budgeter.cost(b), 4), "required": b.required} for b in blocks]
    out = {"query": QUERY, "hybrid": hybrid, "candidates": candidates, "grid": grid,
           "rendered_at_160": budgeter.pack(blocks, 160).render(),
           "search": search_tensors(mem), "trace": deploy_trace()}
    (HERE / "results" / "figures.json").write_text(json.dumps(out, indent=1))
    print(f"{len(hybrid)} memories, {len(candidates)} candidates, {len(grid)} budgets")


if __name__ == "__main__":
    main()
