"""Run the Docker sandbox for real: check each limit, then let the agent fix a bug inside containers.

Every command is a throwaway container: `--rm`, a `ct-c10-` name, `--network none`, memory and
CPU limits. The image has no pytest and the network is off, so the demo copies pytest's
pure-Python packages from this interpreter into the workspace and points PYTHONPATH at them.

    docker pull python:3.12-slim
    python c10-agent-harness/scripts/demo_docker_sandbox.py
"""

from __future__ import annotations

import importlib.util
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from harness import Agent, AgentConfig, DockerSandbox, Limits, ScriptedModel, call, coding_tools  # noqa: E402
from harness.git_tools import init_repo  # noqa: E402
from harness.sandbox import docker_unusable_reason  # noqa: E402

FIX_OLD = "    mid = len(s) // 2\n    return s[mid]"
FIX_NEW = ("    mid = len(s) // 2\n"
           "    if len(s) % 2 == 0:\n"
           "        return (s[mid - 1] + s[mid]) / 2\n"
           "    return s[mid]")
SCRIPT = [
    "Reproduce the failure.\n" + call("run_tests"),
    "Read the code.\n" + call("read_file", path="stats_lib/core.py"),
    "Average the two middle values.\n" + call("edit_file", path="stats_lib/core.py", search=FIX_OLD, replace=FIX_NEW),
    "Verify.\n" + call("run_tests"),
    "Fixed median for even-length lists.",
]
NET_PROBE = ("import socket\ns = socket.socket()\ns.settimeout(2)\ntry:\n    s.connect(('1.1.1.1', 80))\n"
             "    print('connected')\nexcept OSError as e:\n    print('blocked:', e)\n")
FORK_PROBE = ("import os, time\nn = 0\ntry:\n    while n < 100:\n        if os.fork() == 0:\n"
              "            time.sleep(30)\n            os._exit(0)\n        n += 1\n"
              "except OSError as e:\n    print('blocked after', n, 'forks:', e.strerror)\n")


def vendor_pytest(workspace: Path) -> str:
    """Copy pytest and its pure-Python dependencies into the workspace. Returns the PYTHONPATH to use."""
    dest = workspace / ".pydeps"
    dest.mkdir()
    for name in ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "pygments"):
        src = Path(importlib.util.find_spec(name).origin).parent
        shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(importlib.util.find_spec("py").origin, dest / "py.py")  # pytest's one-file shim
    return "/workspace/.pydeps"


def fix_bug_in_docker(workspace: Path) -> list[str]:
    """Agent + scripted model: tools edit the host files, every test run happens in a container."""
    (workspace / ".gitignore").open("a").write(".pydeps/\n")  # keep the copy out of checkpoints
    pythonpath = vendor_pytest(workspace)
    init_repo(workspace)
    box = DockerSandbox(workspace, limits=Limits(memory_mb=512), cpus=2.0, extra_env={"PYTHONPATH": pythonpath})
    agent = Agent(ScriptedModel(SCRIPT), coding_tools(), workspace, sandbox=box, config=AgentConfig(max_steps=8))
    agent.run("The test suite fails on median for even-length lists. Fix it.")
    return [e.data["content"].splitlines()[0] for e in agent.log.events
            if e.type == "ToolResult" and e.data["name"] == "run_tests"]


def check_limits(ws: Path) -> list[tuple[str, str, str]]:
    (ws / "net.py").write_text(NET_PROBE)
    (ws / "fork.py").write_text(FORK_PROBE)
    (ws / "a.txt").write_text("hello\n")
    ws.chmod(0o700)
    box = DockerSandbox(ws, limits=Limits(memory_mb=128), cpus=0.5, pids_limit=16)
    rows = []

    def row(what: str, res, expected: str, shown: str | None = None) -> None:
        got = shown if shown is not None else (res.stdout + res.stderr).strip().splitlines()[-1]
        rows.append((what, expected, f"{got} [exit {res.exit_code}, {res.duration_s:.2f}s]"))

    row("read a mode-700 workspace", box.run("cat a.txt"), "hello")
    res = box.run("echo written > b.txt; id -u")
    row("write as the host user", res, "b.txt owned by this uid",
        f"container uid {res.stdout.strip()}, b.txt uid {(ws / 'b.txt').stat().st_uid}")
    row("network none", box.run("python net.py"), "blocked")
    row("cgroup memory.max", box.run("cat /sys/fs/cgroup/memory.max"), str(128 * 2**20))
    row("cgroup cpu.max", box.run("cat /sys/fs/cgroup/cpu.max"), "50000 100000")
    row("cgroup pids.max", box.run("cat /sys/fs/cgroup/pids.max"), "16")
    row("fork past --pids-limit", box.run("python fork.py"), "blocked")
    res = box.run("python -c \"x = b'x' * (512 * 2**20)\"")
    row("allocate 512 MB under a 128 MB limit", res, "killed_reason=memory", f"killed_reason={res.killed_reason}")
    res = DockerSandbox(ws, limits=Limits(memory_mb=128, cpu_s=1), cpus=0.5).run("python -c 'while True: pass'")
    row("busy loop under --ulimit cpu=1", res, "exit 152 (SIGXCPU)", f"killed_reason={res.killed_reason}")
    res = DockerSandbox(ws, limits=Limits(memory_mb=128, file_size_mb=1), cpus=0.5).run(
        "python -c \"open('big', 'wb').write(b'x' * (5 * 2**20))\"")
    row("write 5 MB under --ulimit fsize=1 MB", res, "at most 1048576 bytes", f"{(ws / 'big').stat().st_size} bytes")
    res = box.run("sleep 30", timeout=2)
    row("sleep 30 with a 2 s timeout", res, "timed out, container removed", f"timed_out={res.timed_out}")
    res = box.run("sleep 30 & echo done")
    row("background child outlives the command", res, "returns at once", res.stdout.strip())
    res = box.run("head -c 5000000 /dev/zero | tr '\\0' x")
    row("5 MB of output", res, "captured in bounded memory", f"{len(res.stdout)} chars kept of {res.stdout_bytes} bytes")
    return rows


def main() -> None:
    if reason := docker_unusable_reason():
        sys.exit(f"cannot run the demo: {reason}")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "limits"
        ws.mkdir()
        rows = check_limits(ws)
        width = max(len(r[0]) for r in rows)
        print("--- limits, each row one fresh container ---")
        for what, expected, got in rows:
            print(f"{what:<{width}}  expected: {expected}\n{'':<{width}}  got:      {got}")

        box = DockerSandbox(ws, limits=Limits(memory_mb=128), cpus=0.5)
        times = [box.run("true").duration_s * 1000 for _ in range(10)]
        print(f"\ncontainer start to exit, `true`, median of 10: {statistics.median(times):.0f} ms")

        repo = Path(tmp) / "toy"
        shutil.copytree(HERE / "examples" / "toy_repo", repo, ignore=shutil.ignore_patterns("__pycache__"))
        print("\n--- agent fixing the toy repo, every run_tests in a container ---")
        for line in fix_bug_in_docker(repo):
            print(line)

    print("\nhardware: AMD EPYC 7R13, 8 threads; image python:3.12-slim")


if __name__ == "__main__":
    main()
