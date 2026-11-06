"""Five intentional failures, each with the symptom you would see.

    python c11-agent-memory/scripts/break_it.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from agentmem import AgentMemory, Block, BudgetError, ContextBudgeter, MemoryEvent, truncate_left  # noqa: E402
from agentmem.bench.deploy_eval import run_deploy  # noqa: E402
from agentmem.bench.deploy_world import make_world  # noqa: E402


def stale_memory() -> None:
    print("1. append-only memory: the old value keeps its place in the ranking")
    for name, mem in (("supersede", AgentMemory()), ("append-only", AgentMemory(supersede=False))):
        mem.observe(MemoryEvent("The deploy region for atlas is eu-west-1.", "user", 1.0))
        mem.observe(MemoryEvent("Heads up: we moved the deploy region for atlas to us-east-2.", "user", 2.0))
        print(f"   {name:<12}", [h.item.text for h in mem.retrieve("deploy region for atlas", 3.0)])


def rumor_overwrites() -> None:
    print("2. no confidence margin: a hedge replaces a confirmed fact")
    for margin in (0.3, 1.0):
        mem = AgentMemory(use_writer=False)
        mem.facts.override_margin = margin
        mem.observe(MemoryEvent("The owner for atlas is priya.", "user", 1.0))
        mem.observe(MemoryEvent("I think the owner for atlas might be chen, not sure.", "user", 2.0))
        print(f"   margin={margin}: current owner = {mem.facts.current('atlas', 'owner').value}")


def budget_too_small() -> None:
    print("3. a budget smaller than the required blocks fails loudly, not silently")
    blocks = [Block("instructions", "You are a deploy agent. " * 10, required=True, compressible=False)]
    try:
        ContextBudgeter().pack(blocks, 20)
    except BudgetError as e:
        print("   BudgetError:", e)


def naive_truncation() -> None:
    print("4. keep-the-newest truncation evicts the instructions first")
    blocks = [Block("events", "SYSTEM: never deploy on Fridays.", order=0)]
    blocks += [Block("events", f"log line {i}: status=200 latency={i}ms", order=i) for i in range(1, 40)]
    kept = truncate_left(blocks, 120)
    print(f"   kept {len(kept)} of {len(blocks)} blocks; instructions kept: {any('SYSTEM' in b.text for b in kept)}")
    print("   same bug in the deploy world, transcript at B=256 vs. no memory at all:")
    for cond in ("full_transcript", "no_memory"):
        rs = []
        for s in range(10):
            rs += run_deploy(make_world(s), cond, 256)
        print(f"   {cond:<16} success {sum(r.success for r in rs) / len(rs):.2f}  mean steps {sum(r.steps for r in rs) / len(rs):.2f}")


def salted_hash() -> None:
    print("5. Python hash() in the embedder: a different bucket in every process")
    for seed in ("1", "2"):
        code = "print(hash('atlas') % 512)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={"PYTHONHASHSEED": seed}).stdout.strip()
        print(f"   PYTHONHASHSEED={seed}: hash('atlas') % 512 = {out}")


if __name__ == "__main__":
    stale_memory()
    rumor_overwrites()
    budget_too_small()
    naive_truncation()
    salted_hash()
