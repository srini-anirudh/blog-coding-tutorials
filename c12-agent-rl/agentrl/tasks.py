"""Task generator for the toy filesystem environment.

A task is: a small filesystem of config files, a sentence that says what to change, and the target filesystem
that a correct agent leaves behind. The five curriculum levels add one difficulty at a time:

    1  single tool        "set port in db/main to 42"                 edit, submit
    2  multiple tools     "set port to 42"                            search, edit, submit
    3  dependent tools    "set port in db/main to value of ttl"       search, read, edit, submit
    4  failure recovery   level 2 or 3 tasks, or a stale path, with injected tool failures
    5  long horizon       two subtasks joined by "and", with injected tool failures

Every file holds three ``key = value`` lines. Values are integers 0-99 so that each one is a single token.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from .failures import FailureConfig

DIRS = ["app", "db", "web", "auth", "cache", "jobs"]
NAMES = ["main", "prod", "dev", "base"]
PATHS = [f"{d}/{n}" for d in DIRS for n in NAMES]
KEYS = ["port", "host", "timeout", "retries", "workers", "level",
        "limit", "ttl", "size", "mode", "rate", "depth"]
VALUES = [str(v) for v in range(100)]
KEYS_PER_FILE = 3

Files = dict[str, dict[str, str]]  # path -> {key: value}


@dataclass(frozen=True)
class Subtask:
    """One required edit. ``kind`` says how much the agent must discover before it can make it."""

    kind: str                 # "direct" | "search" | "dependent" | "stale"
    key: str                  # key to change
    path: str                 # file that really holds ``key``
    value: str                # value it must end up with
    source_key: str | None = None   # dependent: the key whose value must be copied
    source_path: str | None = None  # dependent: the file that holds ``source_key``
    stale_path: str | None = None   # stale: the wrong path written in the task text

    def text(self) -> str:
        if self.kind == "direct":
            return f"set {self.key} in {self.path} to {self.value}"
        if self.kind == "stale":
            return f"set {self.key} in {self.stale_path} to {self.value}"
        if self.kind == "search":
            return f"set {self.key} to {self.value}"
        return f"set {self.key} in {self.path} to value of {self.source_key}"

    def optimal_calls(self) -> int:
        return {"direct": 1, "search": 2, "dependent": 3, "stale": 3}[self.kind]


@dataclass
class Task:
    level: int
    files: Files
    subtasks: list[Subtask]
    failures: FailureConfig = field(default_factory=FailureConfig)
    seed: int = 0

    @property
    def text(self) -> str:
        return " and ".join(s.text() for s in self.subtasks)

    @property
    def optimal_steps(self) -> int:
        """Tool calls on the clean path, plus the final ``submit``."""
        return sum(s.optimal_calls() for s in self.subtasks) + 1

    def target(self) -> Files:
        """The filesystem a correct agent must leave behind."""
        out = copy.deepcopy(self.files)
        for s in self.subtasks:
            out[s.path][s.key] = s.value
        return out


def make_filesystem(rng: random.Random, n_files: int) -> Files:
    paths = rng.sample(PATHS, n_files)
    return {p: {k: rng.choice(VALUES) for k in rng.sample(KEYS, KEYS_PER_FILE)} for p in paths}


def _unique_keys(files: Files) -> list[str]:
    """Keys that appear in exactly one file, so ``search`` returns a single path."""
    counts: dict[str, int] = {}
    for kv in files.values():
        for k in kv:
            counts[k] = counts.get(k, 0) + 1
    return sorted(k for k, c in counts.items() if c == 1)


def _where(files: Files, key: str) -> str:
    return next(p for p, kv in files.items() if key in kv)


def _new_value(rng: random.Random, old: str) -> str:
    return rng.choice([v for v in VALUES if v != old])


def _subtask(kind: str, files: Files, rng: random.Random, used: set[str]) -> Subtask | None:
    """Sample one subtask whose keys are not in ``used``. Returns None when the filesystem cannot host it."""
    unique = [k for k in _unique_keys(files) if k not in used]
    if kind in ("search", "stale"):
        if not unique:
            return None
        key = rng.choice(unique)
        path = _where(files, key)
        value = _new_value(rng, files[path][key])
        if kind == "search":
            return Subtask("search", key, path, value)
        stale = rng.choice([p for p in PATHS if p not in files])
        return Subtask("stale", key, path, value, stale_path=stale)
    if kind == "direct":
        path = rng.choice(sorted(files))
        key = rng.choice([k for k in files[path] if k not in used] or [None])
        if key is None:
            return None
        return Subtask("direct", key, path, _new_value(rng, files[path][key]))
    # dependent: copy the value of a unique key that lives in another file
    for _ in range(20):
        src = rng.choice(unique) if unique else None
        if src is None:
            return None
        src_path = _where(files, src)
        others = [(p, k) for p, kv in files.items() if p != src_path for k in kv
                  if k not in used and k != src and kv[k] != files[src_path][src]]
        if others:
            path, key = rng.choice(others)
            return Subtask("dependent", key, path, files[src_path][src],
                           source_key=src, source_path=src_path)
    return None


LEVEL_KINDS = {
    1: [["direct"]],
    2: [["search"]],
    3: [["dependent"]],
    4: [["search"], ["dependent"], ["stale"]],
    5: [["search", "dependent"], ["dependent", "search"], ["search", "search"], ["dependent", "dependent"]],
}
LEVEL_FAILURES = {
    1: FailureConfig(), 2: FailureConfig(), 3: FailureConfig(),
    4: FailureConfig(p_timeout=0.25, p_tool_error=0.15, p_invalid=0.10),
    5: FailureConfig(p_timeout=0.10, p_tool_error=0.05, p_invalid=0.05),
}


def sample_task(level: int, seed: int) -> Task:
    """Deterministic: the same (level, seed) always gives the same task."""
    rng = random.Random(seed * 7919 + level)
    while True:
        files = make_filesystem(rng, n_files=rng.randint(4, 6))
        kinds = rng.choice(LEVEL_KINDS[level])
        used: set[str] = set()
        subtasks = []
        for kind in kinds:
            s = _subtask(kind, files, rng, used)
            if s is None:
                break
            subtasks.append(s)
            used |= {s.key, s.source_key} - {None}
        if len(subtasks) == len(kinds):
            return Task(level, files, subtasks, LEVEL_FAILURES[level], seed)
