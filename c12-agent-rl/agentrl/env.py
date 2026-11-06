"""The toy filesystem environment: locate the file, edit the value, submit.

State   = filesystem + task + history.
Actions = list DIR | read PATH | search KEY | edit PATH KEY VALUE | submit.
Reward  = 1[final filesystem == target], given once, when the agent submits.

The API is the classic one: ``reset``, ``observe``, ``step``, ``reward``, ``done``. It speaks text so that any
language-model policy can drive it.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from .failures import INVALID, MISSING, inject
from .tasks import VALUES, Files, Task
from .trajectory import Step, Trajectory


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args: tuple[str, ...]
    description: str


TOOLS = {
    "list": ToolSpec("list", ("dir",), "list the files in a directory"),
    "read": ToolSpec("read", ("path",), "print every key = value line of a file"),
    "search": ToolSpec("search", ("key",), "list the files that define a key"),
    "edit": ToolSpec("edit", ("path", "key", "value"), "set key to value in a file"),
    "submit": ToolSpec("submit", (), "finish the episode; the files are checked"),
}


def parse_action(text: str) -> tuple[str, list[str]] | None:
    """'edit db/main port 42' -> ('edit', ['db/main', 'port', '42']). None if malformed."""
    parts = text.split()
    if not parts or parts[0] not in TOOLS or len(parts) - 1 != len(TOOLS[parts[0]].args):
        return None
    return parts[0], parts[1:]


class FileSystemEnv:
    def __init__(self, task: Task, max_steps: int | None = None, seed: int | None = None):
        self.task = task
        self.max_steps = max_steps or task.optimal_steps + 6
        self.seed = task.seed if seed is None else seed

    # -- the environment API -------------------------------------------------------------------------------
    def reset(self) -> str:
        self.files: Files = copy.deepcopy(self.task.files)
        self.rng = random.Random(self.seed)       # drives failure injection only
        self.history: list[Step] = []
        self.revealed: set[tuple[str, str]] = set()  # milestones the agent has observed
        self.submitted = False
        self._obs = self.task.text
        return self._obs

    def observe(self) -> str:
        return self._obs

    def step(self, action: str) -> tuple[str, float, bool]:
        """Run one tool call. Returns (observation, reward, done)."""
        assert not self.done(), "episode is over; call reset()"
        parsed = parse_action(action)
        if parsed is None:
            obs = INVALID
        else:
            command, args = parsed
            obs = inject(self.task.failures, command, self.rng) or self._run(command, args)
        self._obs = obs
        step = Step(action, obs, error=obs.startswith("error"))
        self.history.append(step)  # appended first: done() counts the history
        step.reward = self.reward() if self.done() else 0.0
        step.progress = self.progress()
        return obs, step.reward, self.done()

    def reward(self) -> float:
        """Outcome reward: 1 if the agent submitted and the files equal the target, else 0."""
        return 1.0 if self.submitted and self.files == self.task.target() else 0.0

    def done(self) -> bool:
        return self.submitted or len(self.history) >= self.max_steps

    # -- tools ---------------------------------------------------------------------------------------------
    def _run(self, command: str, args: list[str]) -> str:
        if command == "list":
            hits = sorted(p for p in self.files if p.split("/")[0] == args[0])
            return " ".join(hits) if hits else "error no such dir"
        if command == "read":
            path = args[0]
            if path not in self.files:
                return MISSING
            for key in self.files[path]:
                self.revealed.add((path, key))
            return " ; ".join(f"{k} = {v}" for k, v in self.files[path].items())
        if command == "search":
            hits = sorted(p for p, kv in self.files.items() if args[0] in kv)
            for p in hits:
                self.revealed.add((p, "*"))
            return "found " + " ".join(hits) if hits else "found none"
        if command == "edit":
            path, key, value = args
            if path not in self.files:
                return MISSING
            if key not in self.files[path]:
                return "error no such key"
            if value not in VALUES:
                return INVALID
            self.files[path][key] = value
            return "ok"
        self.submitted = True  # submit
        return "submitted"

    # -- milestones, used for shaped rewards ---------------------------------------------------------------
    def milestones(self) -> list[bool]:
        """Each subtask has a 'located' milestone and an 'edited' milestone; dependent ones also 'read'."""
        out = []
        for s in self.task.subtasks:
            if s.kind == "dependent":
                out.append((s.source_path, "*") in self.revealed or (s.source_path, s.source_key) in self.revealed)
                out.append((s.source_path, s.source_key) in self.revealed)
            elif s.kind in ("search", "stale"):
                out.append((s.path, "*") in self.revealed or (s.path, s.key) in self.revealed)
            out.append(self.files[s.path][s.key] == s.value)
        return out

    def progress(self) -> float:
        """Potential Phi(s) in [0, 1]. Defined as 0 once the episode is over, so that shaping telescopes."""
        if self.done():
            return 0.0
        m = self.milestones()
        return sum(m) / len(m)


def run_episode(env: FileSystemEnv, agent, forced: list[str] | None = None) -> Trajectory:
    """Text-level rollout. ``agent(observation) -> action``; ``forced`` actions are played first."""
    obs = env.reset()
    forced = list(forced or [])
    while not env.done():
        if forced:
            action, is_forced = forced.pop(0), True
        else:
            action, is_forced = agent(obs), False
        obs, _, _ = env.step(action)
        env.history[-1].forced = is_forced
    return Trajectory(env.task.text, env.task.level, env.task.seed, steps=list(env.history),
                      success=env.reward() == 1.0, submitted=env.submitted, truncated=not env.submitted)
