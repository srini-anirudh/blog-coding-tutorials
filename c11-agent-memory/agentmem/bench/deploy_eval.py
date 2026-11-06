"""Run the deploy world under the four memory conditions with a fixed rule policy.

The policy reads only the context string. It understands three kinds of line:

    [s3t2] `deploy cobalt` -> AssetError          a step (current task, or the transcript)
    episode s1t2 (session 1): `deploy cobalt` -> AssetError; `build_assets cobalt` -> ok; ...
    rule: when `deploy {entity}` fails with AssetError, run `build_assets {entity}` and retry

So it can only use the past if the past is in its context. What the context contains
is the one thing that differs between conditions.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from ..budget import Block, ContextBudgeter, truncate_left
from ..consolidate import error_signature
from ..memory import AgentMemory
from ..records import Episode
from ..tokens import count_tokens
from .deploy_world import FIX_TOOLS, QUIRKS, DeployTask, DeployWorld

INSTRUCTIONS = ("You are a deploy agent. Tools: deploy, migrate, login, build_assets, clear_cache; each takes "
                "a project name. Deploy the requested project. If a deploy fails, fix the cause and retry.")
CONDITIONS = ("no_memory", "full_transcript", "retrieved", "oracle")

_STEP = re.compile(r"`([a-z_]+) ([a-z]+)` -> (\w+)")
_TAGGED = re.compile(r"^\[(?P<tid>s\d+t\d+)\]")
_EPISODE = re.compile(r"episode (?P<tid>s\d+t\d+)")
_RULE = re.compile(r"when `deploy \{entity\}` fails with (?P<err>\w+), run `(?P<fix>\w+) \{entity\}`")


def parse_steps(context: str) -> dict[str, list[tuple[str, str, str]]]:
    """{task id: [(tool, project, result), ...]} from step lines and episode lines."""
    steps: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for line in context.splitlines():
        m = _TAGGED.match(line) or _EPISODE.search(line)
        if m:
            steps[m["tid"]] += _STEP.findall(line)
    return steps


def recovery_fix(trajectory: list[tuple[str, str, str]]) -> tuple[str, str] | None:
    """(error, fix tool) if the trajectory shows a failed deploy later fixed and retried."""
    err = None
    fix = None
    for tool, _, result in trajectory:
        if tool == "deploy" and result != "ok":
            err = result
        elif tool != "deploy" and err:
            fix = tool
        elif tool == "deploy" and result == "ok" and err and fix:
            return err, fix
    return None


class DeployPolicy:
    """Deterministic: known fix for this project > rule for this error > analogy > explore.

    Exploration tries fixes that past episodes showed useless for this project last.
    With `analogy=False` the policy is literal: it uses episodes of the same project
    and explicit rules, but does not generalise from another project's episode.
    """

    def __init__(self, analogy: bool = True) -> None:
        self.analogy = analogy

    def __call__(self, context: str, task_id: str, project: str) -> str:
        steps = parse_steps(context)
        mine = steps.pop(task_id, [])
        rules = {m["err"]: m["fix"] for m in _RULE.finditer(context)}
        own, analog, useless = None, {}, set()
        for traj in steps.values():
            rec = recovery_fix(traj)
            if rec and traj[0][1] == project:
                own = rec[1]
            elif rec and self.analogy:
                analog.setdefault(rec[0], rec[1])
            if traj and traj[0][1] == project:  # fixes that were followed by another failure
                useless |= {a[0] for a, b in zip(traj, traj[1:]) if a[0] != "deploy" and b[2] != "ok"}
        tried = {tool for tool, _, _ in mine}
        if not mine:
            return f"{own} {project}" if own else f"deploy {project}"
        last_tool, _, last_result = mine[-1]
        if last_tool != "deploy":
            return f"deploy {project}"
        explore = [f for f in FIX_TOOLS if f not in useless] + [f for f in FIX_TOOLS if f in useless]
        for fix in [own, rules.get(last_result), analog.get(last_result), *explore]:
            if fix and fix not in tried:
                return f"{fix} {project}"
        return f"deploy {project}"


@dataclass
class TaskResult:
    task: DeployTask
    condition: str
    success: bool
    steps: int
    tokens: float      # mean context tokens per model call
    seen_before: bool  # this project had a task in an earlier session


def step_line(task_id: str, action: str, observation: str) -> str:
    return f"[{task_id}] `{action}` -> {error_signature(observation) or 'ok'}"


def _pack(budgeter: ContextBudgeter, request: str, events: list[str], obs: str | None, budget: int,
          memories: list[str] = ()) -> str:
    blocks = [Block("instructions", INSTRUCTIONS, required=True, compressible=False),
              Block("request", request, required=True, compressible=False)]
    blocks += [Block("memory", m, value=1.0, order=i) for i, m in enumerate(memories)]
    blocks += [Block("events", e, value=0.5 + i / max(1, len(events)), order=i) for i, e in enumerate(events)]
    if obs:
        blocks.append(Block("observation", obs, value=0.9))
    return budgeter.pack(blocks, budget).render()


def oracle_line(world: DeployWorld, project: str) -> str:
    err = world.quirks[project]
    if err is None:
        return f"episode s0t0 (oracle): `deploy {project}` -> ok"
    fix = QUIRKS[err][0]
    return f"episode s0t0 (oracle): `deploy {project}` -> {err}; `{fix} {project}` -> ok; `deploy {project}` -> ok"


def run_deploy(world: DeployWorld, condition: str, budget: int, max_steps: int = 8,
               policy: Callable[[str, str, str], str] | None = None,
               memory: AgentMemory | None = None) -> list[TaskResult]:
    policy = policy or DeployPolicy()
    memory = memory or AgentMemory()
    budgeter = ContextBudgeter()
    transcript: list[Block] = []
    results, seen, session = [], set(), 0
    for task in world.tasks:
        if task.session != session:
            world.new_session()
            memory.end_session()
            session = task.session
        request = f"({task.id}) Deploy {task.project}."
        events: list[str] = []
        obs: str | None = None
        tokens, success = [], False
        for step in range(max_steps):
            t = task.session + (len(memory.episodes) + 1) / 1000
            if condition == "full_transcript":
                fixed = count_tokens(INSTRUCTIONS) + count_tokens(request) + 12
                kept = truncate_left(transcript, budget - fixed)
                ctx = f"## Instructions\n{INSTRUCTIONS}\n\n## Transcript\n" + "\n".join(b.text for b in kept) + \
                    f"\n\n## Request\n{request}"
            elif condition == "retrieved":
                err = error_signature(obs or "") or ""
                ctx = memory.build_context(INSTRUCTIONS, request, now=t, budget=budget, recent_events=events,
                                           observation=obs, query=f"deploy {task.project} {err}").render()
            else:
                mem = [oracle_line(world, task.project)] if condition == "oracle" else []
                ctx = _pack(budgeter, request, events, obs, budget, mem)
            tokens.append(count_tokens(ctx))
            action = policy(ctx, task.id, task.project)
            obs, reward = world.step(action)
            line = step_line(task.id, action, obs)
            events.append(line)
            transcript.append(Block("events", f"{line}\n{obs}", order=len(transcript)))
            memory.record_step(Episode(f"deploy {task.project}", action, obs, reward, t, task.id,
                                       task.session, task.project))
            if reward > 0 and action.startswith("deploy"):
                success = True
                break
        memory.end_task(task.id)
        results.append(TaskResult(task, condition, success, len(events), sum(tokens) / len(tokens),
                                  task.project in seen))
        seen.add(task.project)
    return results


def summarise(results: list[TaskResult]) -> dict[str, dict[str, float]]:
    by: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        by[r.condition].append(r)
    out = {}
    for cond, rs in by.items():
        seen = [r for r in rs if r.seen_before]
        new = [r for r in rs if not r.seen_before]
        out[cond] = {
            "success": sum(r.success for r in rs) / len(rs),
            "steps": sum(r.steps for r in rs) / len(rs),
            "steps_seen": sum(r.steps for r in seen) / max(1, len(seen)),
            "steps_new": sum(r.steps for r in new) / max(1, len(new)),
            "tokens": sum(r.tokens for r in rs) / len(rs),
            "n": len(rs),
        }
    return out
