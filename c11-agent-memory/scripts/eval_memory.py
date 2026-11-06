"""The memory evaluation: four conditions, a budget sweep, ablations, causal checks.

    python c11-agent-memory/scripts/eval_memory.py [--seeds 20]

Writes results/eval_memory.json and prints every table quoted in the post and README.
Rule reader and rule policy, so the whole run is deterministic and takes about a minute.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from agentmem import AgentMemory, HybridWeights  # noqa: E402
from agentmem.bench import deploy_eval, facts_eval  # noqa: E402
from agentmem.bench.deploy_world import make_world  # noqa: E402
from agentmem.bench.facts_eval import CorruptedMemory, run_facts  # noqa: E402
from agentmem.bench.facts_world import make_script  # noqa: E402
from agentmem.bench.hardware import hardware_label, use_threads  # noqa: E402

use_threads(8)


def ci95(per_seed: list[float], draws: int = 2000) -> tuple[float, float]:
    """Bootstrap over seeds: whole scripts are the independent unit, not single probes."""
    rng = random.Random(0)
    means = sorted(sum(rng.choices(per_seed, k=len(per_seed))) / len(per_seed) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def facts_table(seeds: int, budgets: list[int]) -> dict:
    out: dict = {}
    for b in budgets:
        scripts = [make_script(s) for s in range(seeds)]
        rows = facts_eval.summarise(run_facts(scripts, b))
        for cond, row in rows.items():
            per_seed = [facts_eval.summarise(run_facts([scripts[s]], b, conditions=(cond,)))[cond]["accuracy"]
                        for s in range(min(seeds, 20))]
            row["ci95"] = ci95(per_seed)
        out[str(b)] = rows
        print(f"\nFACTS  budget B={b}  (probes per condition: {rows['oracle']['n']})")
        print(f"{'condition':<18}{'accuracy':>9}{'95% CI':>15}{'tokens':>8}")
        for cond, r in rows.items():
            print(f"{cond:<18}{r['accuracy']:>9.3f}   [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]{r['tokens']:>8.0f}")
    return out


def facts_kinds(rows: dict) -> None:
    kinds = ["stated", "updated", "rumor", "deleted", "today", "after_today", "unseen"]
    print("\nFACTS  accuracy by probe kind, B=1024")
    print(f"{'condition':<18}" + "".join(f"{k:>12}" for k in kinds))
    for cond, r in rows.items():
        print(f"{cond:<18}" + "".join(f"{r.get(k, float('nan')):>12.2f}" for k in kinds))


def facts_ablations(seeds: int, budget: int) -> dict:
    def broken_margin() -> AgentMemory:
        m = AgentMemory(use_writer=False, scope_gate=False)
        m.facts.override_margin = 1.0
        return m

    variants = {
        "full system": AgentMemory,
        "append-only (no SUPERSEDE)": lambda: AgentMemory(supersede=False),
        "no writer, no scope gate": lambda: AgentMemory(use_writer=False, scope_gate=False),
        "no writer, no margin": broken_margin,
        "recency only (a=0,b=1,g=0)": lambda: AgentMemory(weights=HybridWeights(0, 1, 0)),
        "semantic only (a=1,b=0,g=0)": lambda: AgentMemory(weights=HybridWeights(1, 0, 0)),
        "corrupted values": CorruptedMemory,
    }
    scripts = [make_script(s) for s in range(seeds)]
    out = {}
    print(f"\nFACTS  ablations of the retrieved condition, B={budget}")
    print(f"{'variant':<30}{'accuracy':>9}{'updated':>9}{'rumor':>7}{'today+1':>9}{'stated':>8}{'tokens':>8}")
    for name, factory in variants.items():
        r = facts_eval.summarise(run_facts(scripts, budget, memory_factory=factory, conditions=("retrieved",)))["retrieved"]
        out[name] = r
        print(f"{name:<30}{r['accuracy']:>9.3f}{r.get('updated', 0):>9.2f}{r.get('rumor', 0):>7.2f}"
              f"{r.get('after_today', 0):>9.2f}{r.get('stated', 0):>8.2f}{r['tokens']:>8.0f}")
    return out


def deploy_table(seeds: int, budgets: list[int]) -> dict:
    out: dict = {}
    for b in budgets:
        res = []
        for s in range(seeds):
            for cond in deploy_eval.CONDITIONS:
                res += deploy_eval.run_deploy(make_world(s), cond, b)
        rows = deploy_eval.summarise(res)
        out[str(b)] = rows
        print(f"\nDEPLOY  budget B={b}  (tasks per condition: {rows['oracle']['n']})")
        print(f"{'condition':<18}{'success':>9}{'steps':>7}{'seen':>7}{'new':>7}{'tokens':>8}")
        for cond, r in rows.items():
            print(f"{cond:<18}{r['success']:>9.3f}{r['steps']:>7.2f}{r['steps_seen']:>7.2f}{r['steps_new']:>7.2f}{r['tokens']:>8.0f}")
    return out


def deploy_rules(seeds: int, budget: int) -> dict:
    """Does consolidation help? Depends on whether the reader can generalise by itself."""
    out = {}
    print(f"\nDEPLOY  consolidation ablation, retrieved condition, B={budget}")
    print(f"{'reader':<10}{'memory':<14}{'success':>9}{'steps':>7}{'steps on new projects':>24}")
    for reader, analogy in (("analogy", True), ("literal", False)):
        for mem, support in (("rules", 2), ("no rules", 10**6)):
            res = []
            for s in range(seeds):
                res += deploy_eval.run_deploy(make_world(s), "retrieved", budget,
                                              policy=deploy_eval.DeployPolicy(analogy=analogy),
                                              memory=AgentMemory(consolidate_min_support=support))
            r = deploy_eval.summarise(res)["retrieved"]
            out[f"{reader}/{mem}"] = r
            print(f"{reader:<10}{mem:<14}{r['success']:>9.3f}{r['steps']:>7.2f}{r['steps_new']:>24.2f}")
    return out


def step_limit(seeds: int, budget: int) -> dict:
    """The success column depends on the step limit, so read it next to the steps column."""
    out = {}
    print(f"\nDEPLOY  step limit, budget B={budget}")
    print(f"{'max steps':<11}{'no_memory':>11}{'retrieved':>11}")
    for limit in (4, 6, 8, 9, 12):
        res = []
        for s in range(seeds):
            for cond in ("no_memory", "retrieved"):
                res += deploy_eval.run_deploy(make_world(s), cond, budget, max_steps=limit)
        rows = deploy_eval.summarise(res)
        out[str(limit)] = {c: rows[c]["success"] for c in rows}
        print(f"{limit:<11}{rows['no_memory']['success']:>11.3f}{rows['retrieved']['success']:>11.3f}")
    return out


def harness_check(seeds: int, budgets: list[int]) -> dict:
    """The same policy through the real C10 Agent. B here also pays for the tool schemas."""
    try:
        from agentmem.bench.deploy_harness import run_deploy_harness
    except ImportError:
        print("\nHARNESS  skipped: ../c10-agent-harness is not present")
        return {}
    out: dict = {}
    for b in budgets:
        res = []
        for s in range(seeds):
            for cond in ("no_memory", "retrieved", "oracle"):
                res += run_deploy_harness(make_world(s), cond, b)
        rows = deploy_eval.summarise(res)
        out[str(b)] = rows
        print(f"\nHARNESS  harness.Agent loop, budget B={b}")
        print(f"{'condition':<18}{'success':>9}{'steps':>7}{'seen':>7}{'new':>7}{'prompt tokens':>15}")
        for cond, r in rows.items():
            print(f"{cond:<18}{r['success']:>9.3f}{r['steps']:>7.2f}{r['steps_seen']:>7.2f}{r['steps_new']:>7.2f}{r['tokens']:>15.0f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    t0 = time.perf_counter()
    budgets = [256, 512, 1024, 2048, 4096]
    out = {"seeds": args.seeds, "hardware": hardware_label()}
    out["facts"] = facts_table(args.seeds, budgets)
    facts_kinds(out["facts"]["1024"])
    out["facts_ablations"] = facts_ablations(args.seeds, 1024)
    out["deploy"] = deploy_table(args.seeds, budgets)
    out["deploy_rules"] = deploy_rules(args.seeds, 1024)
    out["step_limit"] = step_limit(args.seeds, 1024)
    out["harness"] = harness_check(args.seeds, [1024, 2048])
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "eval_memory.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nwall time {time.perf_counter() - t0:.0f}s on {hardware_label()}")


if __name__ == "__main__":
    main()
