"""A multi-session deploy world where past episodes save future steps.

Each project has at most one quirk: its deploy fails with a specific error until a
specific fix has run in the current session (tokens expire, schemas drift). The error
text does not name the fix. An agent with no memory has to explore fixes one by one.
An agent that remembers a past episode for this project fixes it before deploying. An
agent with a consolidated rule fixes any project with that error after one failure,
including projects it has never seen.

Tools (each takes a project name): deploy, migrate, login, build_assets, clear_cache.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

QUIRKS: dict[str, tuple[str, str]] = {
    # error class -> (fix tool, message)
    "MigrationError": ("migrate", "MigrationError: schema version {a} does not match code version {b}"),
    "AuthError": ("login", "AuthError: registry rejected the push token (401)"),
    "AssetError": ("build_assets", "AssetError: bundle manifest missing from build output"),
}
FIX_TOOLS = ["clear_cache", "build_assets", "login", "migrate"]   # exploration order without memory
PROJECTS = ["atlas", "borealis", "cobalt", "dune", "ember", "fjord", "granite", "harbor", "iris", "juniper"]


def _deploy_log(rng: random.Random, project: str, tail: str) -> str:
    """A deploy log: mostly progress noise, the signal in the last line."""
    lines = [f"$ deploy {project}"]
    for i, stage in enumerate(["resolve", "fetch", "compile", "test", "package", "push", "rollout"], 1):
        lines.append(f"[{i}/7] {stage:<8} {project}: done in {rng.randint(1, 90)}.{rng.randint(0, 9)}s "
                     f"(cache {'hit' if rng.random() < 0.6 else 'miss'})")
    lines.append(tail)
    return "\n".join(lines)


@dataclass
class DeployTask:
    id: str            # "s3t2"
    session: int
    project: str


@dataclass
class DeployWorld:
    quirks: dict[str, str | None]                # project -> error class or None
    tasks: list[DeployTask] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed + 7919)
        self.fixed: set[tuple[str, str]] = set()  # (project, fix) done this session

    def new_session(self) -> None:
        self.fixed.clear()

    def step(self, action: str) -> tuple[str, float]:
        """Run 'tool project'. Returns (observation, reward): +1 deployed, -1 failed, 0 otherwise."""
        tool, _, project = action.partition(" ")
        if project not in self.quirks:
            return f"Error: unknown project {project!r}", -1.0
        if tool == "deploy":
            err = self.quirks[project]
            if err and (project, QUIRKS[err][0]) not in self.fixed:
                msg = QUIRKS[err][1].format(a=self.rng.randint(30, 60), b=self.rng.randint(61, 90))
                return _deploy_log(self.rng, project, msg), -1.0
            return _deploy_log(self.rng, project, f"deployed {project}: healthy"), 1.0
        if tool in FIX_TOOLS:
            self.fixed.add((project, tool))
            return f"{tool} {project}: ok", 0.0
        return f"Error: unknown tool {tool!r}", -1.0


def make_world(seed: int, n_sessions: int = 8, tasks_per_session: int = 3, n_projects: int = 10,
               new_from_session: int = 5) -> DeployWorld:
    """Projects 0..5 appear from session 1; the rest only from `new_from_session` on."""
    rng = random.Random(seed)
    projects = rng.sample(PROJECTS, n_projects)
    # Every error occurs on two early projects (so a rule can be consolidated) and
    # once more on a late project (so the rule can be tested on a project never seen).
    pattern = ["MigrationError", "AuthError", "AssetError"] * 3 + [None]
    quirks = dict(zip(projects, pattern))
    early, late = projects[:6], projects[6:]
    tasks = []
    for s in range(1, n_sessions + 1):
        pool = early if s < new_from_session else early + late * 2   # late projects are favoured once they exist
        for j in range(tasks_per_session):
            tasks.append(DeployTask(f"s{s}t{j + 1}", s, rng.choice(pool)))
    return DeployWorld(quirks, tasks, seed)
