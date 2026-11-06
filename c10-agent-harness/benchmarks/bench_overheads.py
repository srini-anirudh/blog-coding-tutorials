"""What the harness itself costs: sandbox start-up, search backends, indexing, per-step overhead.

    python c10-agent-harness/benchmarks/bench_overheads.py
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from harness import Agent, AgentConfig, ContextBuilder, LocalSandbox, ScriptedModel, ToolRegistry, call, tool  # noqa: E402
from harness.git_tools import init_repo  # noqa: E402
from harness.repomap import build_repo_map  # noqa: E402
from harness.search import SymbolIndex, iter_files, search_gitgrep, search_python  # noqa: E402
from harness.sandbox import DockerSandbox, Limits, docker_unusable_reason, run_process  # noqa: E402
from harness.tokens import count_tokens  # noqa: E402


@tool
def noop() -> str:
    """Do nothing."""
    return "ok"


def timed(fn, reps: int = 20) -> float:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000


def main() -> None:
    print("--- running `true` (median of 20, ms) ---")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        rows = {
            "subprocess.run": lambda: subprocess.run(["/usr/bin/true"]),
            "run_process (group, capture)": lambda: run_process(["/usr/bin/true"], ws, {}, 10.0),
            "LocalSandbox, no network wrap": lambda: LocalSandbox(ws, Limits(network=True)).run("true"),
            "LocalSandbox, network denied": lambda: LocalSandbox(ws).run("true"),
        }
        for name, fn in rows.items():
            print(f"{name:<34}{timed(fn):>8.1f}")
        # A fresh container per command. Skipped, with the reason, on a machine that cannot run one.
        if reason := docker_unusable_reason():
            print(f"{'DockerSandbox, fresh container':<34}skipped: {reason}")
        else:
            print(f"{'DockerSandbox, fresh container':<34}{timed(lambda: DockerSandbox(ws).run('true'), 5):>8.1f}")

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        n = 0
        for f in ROOT.glob("c*/**/*.py"):
            if ".venv" in f.parts or "__pycache__" in f.parts:
                continue
            dest = repo / f.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest)
            n += 1
        init_repo(repo)
        lines = sum(len(p.read_text(errors="replace").splitlines()) for p in iter_files(repo, "*.py"))
        print(f"\n--- search over this repo's Python code: {n} files, {lines} lines (ms) ---")
        for name, fn in {"git grep": search_gitgrep, "pure Python": search_python}.items():
            found = len(fn(repo, r"def forward"))
            print(f"{name:<34}{timed(lambda: fn(repo, r'def forward'), 10):>8.1f}   ({found} matches)")
        t = timed(lambda: SymbolIndex(repo), 3)
        idx = SymbolIndex(repo)
        print(f"{'symbol index build':<34}{t:>8.1f}   "
              f"({sum(len(f.symbols) for f in idx.files.values())} symbols)")
        print(f"{'definition lookup':<34}{timed(lambda: idx.definitions('forward'), 20):>8.2f}   "
              f"({len(idx.definitions('forward'))} definitions)")
        print(f"\n--- repo map size against budget ---")
        full = "\n".join(str(p.relative_to(repo)) for p in iter_files(repo, "*.py"))
        raw = sum(count_tokens(p.read_text(errors="replace")) for p in iter_files(repo, "*.py"))
        print(f"{'all source code':<34}{raw:>8} tokens")
        print(f"{'file names only':<34}{count_tokens(full):>8} tokens")
        for budget in (512, 2048, 8192, 10**7):
            m = build_repo_map(repo, budget)
            label = "unbudgeted map" if budget == 10**7 else f"map at budget {budget}"
            print(f"{label:<34}{count_tokens(m):>8} tokens")

    with tempfile.TemporaryDirectory() as tmp:
        steps = 200
        script = [call("noop") for _ in range(steps)] + ["done"]
        agent = Agent(ScriptedModel(script), ToolRegistry([noop]), tmp, context=ContextBuilder(),
                      config=AgentConfig(max_steps=steps + 1, checkpoint_on_start=False))
        t0 = time.perf_counter()
        agent.run("spin")
        total = time.perf_counter() - t0
        print(f"\n--- harness overhead, scripted model and a no-op tool ---")
        print(f"{steps} steps in {total:.3f}s: {total / steps * 1000:.2f} ms per step")


if __name__ == "__main__":
    main()
