"""Step 8, for real: every limit of the Docker sandbox, checked inside actual containers.

The tests that start containers use the `docker_ready` fixture, so they skip with a reason
where Docker is missing or unusable. Every container is throwaway: `--rm`, a `ct-c10-` name,
`--network none`, memory and CPU limits. The fixture below only ever looks at names it made.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest
from helpers import toy_workspace

from harness.fs_tools import FileTracker
from harness.planning import Plan
from harness.sandbox import DockerSandbox, Limits, docker_unusable_reason
from harness.shell import format_exec, run_command
from harness.tools import ToolContext, ToolError
from harness.verify import run_tests

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def containers_named(prefix: str) -> list[str]:
    out = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^/{prefix}", "--format", "{{.Names}}"],
                         capture_output=True, text=True).stdout
    return out.split()


@pytest.fixture
def make_box(docker_ready, tmp_path):
    """Sandboxes with a prefix unique to this test. Teardown proves nothing was left behind."""
    prefix = f"ct-c10-test-{uuid.uuid4().hex[:8]}-"

    def make(**kw) -> DockerSandbox:
        kw.setdefault("limits", Limits(memory_mb=256))
        kw.setdefault("cpus", 1.0)
        return DockerSandbox(tmp_path, name_prefix=prefix, **kw)

    make.prefix = prefix
    yield make
    deadline = time.monotonic() + 5
    while (left := containers_named(prefix)) and time.monotonic() < deadline:
        time.sleep(0.1)  # --rm finishes just after the CLI exits
    for name in left:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    assert not left, f"containers left behind: {left}"


# ---------------------------------------------------------------- argv, no daemon needed

def test_argv_is_throwaway_and_named_for_this_tutorial(tmp_path):
    argv = DockerSandbox(tmp_path).docker_argv("true")
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--name") + 1].startswith("ct-c10-")
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


def test_argv_carries_the_remaining_limits(tmp_path):
    box = DockerSandbox(tmp_path, limits=Limits(open_files=64, cpu_s=7, file_size_mb=2), extra_env={"A": "1"})
    joined = " ".join(box.docker_argv("true"))
    assert "--ulimit nofile=64" in joined and "--ulimit cpu=7:12" in joined
    assert f"--ulimit fsize={2 * 2**20}" in joined and "-e A=1" in joined


@pytest.mark.parametrize("cwd,workdir", [(".", "/workspace"), ("src", "/workspace/src"), ("a/../b", "/workspace/b"),
                                         ("v1.", "/workspace/v1.")])
def test_argv_workdir(tmp_path, cwd, workdir):
    argv = DockerSandbox(tmp_path).docker_argv("true", cwd=cwd)
    assert argv[argv.index("-w") + 1] == workdir


@pytest.mark.parametrize("cwd", ["..", "../x", "a/../../x"])
def test_argv_refuses_a_cwd_outside_the_workspace(tmp_path, cwd):
    with pytest.raises(ValueError, match="escapes"):
        DockerSandbox(tmp_path).docker_argv("true", cwd=cwd)


class _FakeDocker:
    """Answers `docker ...` calls by subcommand, so the skip reasons can be tested on any machine."""

    def __init__(self, failing: str | None = None, hangs: str | None = None):
        self.failing, self.hangs, self.calls = failing, hangs, []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[1] == self.hangs:
            raise subprocess.TimeoutExpired(argv, 30)
        code = 1 if argv[1] == self.failing else 0
        return subprocess.CompletedProcess(argv, code, "", "permission denied on socket\n" if code else "")


def _patch(monkeypatch, fake=None, installed=True):
    monkeypatch.setattr("harness.sandbox.shutil.which", lambda _: "/bin/docker" if installed else None)
    if fake:
        monkeypatch.setattr("harness.sandbox.subprocess.run", fake)


def test_reason_when_docker_is_not_installed(monkeypatch):
    _patch(monkeypatch, installed=False)
    assert docker_unusable_reason() == "docker is not installed"


@pytest.mark.parametrize("failing,expected", [("info", "daemon is not usable"), ("image", "docker pull"),
                                              ("run", "cannot start")])
def test_reason_names_the_failing_check(monkeypatch, failing, expected):
    _patch(monkeypatch, _FakeDocker(failing=failing))
    reason = docker_unusable_reason()
    assert expected in reason and "permission denied on socket" in reason


def test_reason_when_the_daemon_hangs(monkeypatch):
    _patch(monkeypatch, _FakeDocker(hangs="info"))
    assert "no answer" in docker_unusable_reason()


def test_usable_machine_has_no_reason_and_the_probe_is_throwaway(monkeypatch):
    fake = _FakeDocker()
    _patch(monkeypatch, fake)
    assert docker_unusable_reason() is None
    probe = next(a for a in fake.calls if a[1] == "run")
    assert "--rm" in probe and probe[probe.index("--name") + 1].startswith("ct-c10-")
    assert not any(a[1] == "pull" for a in fake.calls)  # an offline test run stays offline


# ---------------------------------------------------------------- real containers

def test_mode_700_workspace_is_readable_and_writable(make_box, tmp_path):
    """The bug this suite first caught: root with no capabilities cannot enter a user's private directory."""
    tmp_path.chmod(0o700)
    (tmp_path / "a.txt").write_text("hello")
    res = make_box().run("cat a.txt; echo; echo written > b.txt; mkdir sub; id -u")
    assert res.ok, res.stderr
    assert res.stdout.split() == ["hello", str(os.getuid())]
    assert (tmp_path / "b.txt").stat().st_uid == os.getuid()
    assert (tmp_path / "sub").stat().st_uid == os.getuid()


def test_exit_code_and_streams(make_box):
    res = make_box().run("echo out; echo err >&2; exit 3")
    assert (res.exit_code, res.stdout, res.stderr) == (3, "out\n", "err\n")
    assert not res.ok and not res.timed_out and res.killed_reason is None


def test_cwd(make_box, tmp_path):
    (tmp_path / "sub").mkdir()
    box = make_box()
    assert box.run("pwd", cwd="sub").stdout.strip() == "/workspace/sub"
    with pytest.raises(ValueError, match="escapes"):
        box.run("pwd", cwd="..")
    with pytest.raises(ValueError, match="not a directory"):
        box.run("pwd", cwd="missing")
    assert not (tmp_path / "missing").exists()


def test_shell_tool_runs_in_the_container(make_box, tmp_path):
    ctx = ToolContext(tmp_path, make_box(), FileTracker(), Plan())
    out = run_command(ctx, "pwd; exit 2")
    assert out.is_error and out.data["exit_code"] == 2 and "/workspace" in out.content
    with pytest.raises(ToolError, match="escapes"):
        run_command(ctx, "pwd", cwd="../..")


def test_host_environment_does_not_leak(make_box, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    out = make_box(extra_env={"MY_FLAG": "on"}).run("env").stdout
    assert "sk-secret" not in out and "HOME=/workspace" in out and "MY_FLAG=on" in out


def test_network_is_none(make_box, tmp_path):
    (tmp_path / "net.py").write_text("import socket\ns = socket.socket()\ns.settimeout(2)\n"
                                     "try:\n    s.connect(('1.1.1.1', 80))\n    print('connected')\n"
                                     "except OSError as e:\n    print('blocked', e)\n")
    assert "blocked" in make_box().run("python net.py").stdout


def test_cgroup_holds_the_configured_limits(make_box):
    box = make_box(limits=Limits(memory_mb=128), cpus=0.5, pids_limit=32)
    res = box.run("cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max /sys/fs/cgroup/pids.max")
    if not res.ok:
        pytest.skip(f"this host does not use cgroup v2: {res.stderr.strip()}")
    assert res.stdout.split() == [str(128 * 2**20), "50000", "100000", "32"]


def test_memory_limit_kills_and_is_reported(make_box, tmp_path):
    res = make_box(limits=Limits(memory_mb=64, timeout_s=30)).run("python -c \"x = b'x' * (512 * 2**20)\"")
    assert res.exit_code == 137 and res.killed_reason == "memory" and not res.timed_out
    assert "memory limit exceeded" in format_exec(res)


def test_pids_limit_stops_a_fork_loop(make_box, tmp_path):
    (tmp_path / "fork.py").write_text("import os, time\nn = 0\ntry:\n    while n < 100:\n"
                                      "        if os.fork() == 0:\n            time.sleep(30)\n            os._exit(0)\n"
                                      "        n += 1\nexcept OSError:\n    print('blocked', n)\n")
    out = make_box(pids_limit=16).run("python fork.py").stdout.split()
    assert out[0] == "blocked" and int(out[1]) < 16


def test_cpu_ulimit_kills_a_busy_loop_and_is_not_called_memory(make_box):
    res = make_box(limits=Limits(memory_mb=128, cpu_s=1, timeout_s=30)).run("python -c 'while True: pass'")
    assert res.exit_code == 152 and res.killed_reason is None and not res.timed_out


def test_file_size_ulimit(make_box, tmp_path):
    box = make_box(limits=Limits(memory_mb=128, file_size_mb=1))
    res = box.run("python -c \"open('big', 'wb').write(b'x' * (5 * 2**20))\"")
    assert res.exit_code != 0 and (tmp_path / "big").stat().st_size <= 2**20


def test_open_files_ulimit(make_box):
    assert make_box(limits=Limits(memory_mb=128, open_files=128)).run("ulimit -n").stdout.strip() == "128"


def test_timeout_removes_the_container(make_box):
    box = make_box()
    results = []
    worker = threading.Thread(target=lambda: results.append(box.run("sleep 30", timeout=3)))
    t0 = time.monotonic()
    worker.start()
    seen = []
    while not seen and time.monotonic() - t0 < 2.5:  # the filter must really see it, or the check below proves nothing
        seen = containers_named(make_box.prefix)
    worker.join()
    res = results[0]
    assert seen, "the container was never visible by its name"
    assert res.timed_out and res.exit_code is None and res.killed_reason == "timeout"
    assert time.monotonic() - t0 < 15


def test_background_child_does_not_hang_the_call(make_box):
    t0 = time.monotonic()
    res = make_box().run("sleep 30 & echo done", timeout=20)
    assert res.ok and res.stdout.strip() == "done" and time.monotonic() - t0 < 10


def test_giant_output_is_captured_in_bounded_memory(make_box):
    box = make_box(limits=Limits(memory_mb=128, capture_bytes=100_000))
    res = box.run("head -c 5000000 /dev/zero | tr '\\0' x")
    assert res.ok and res.stdout_bytes == 5_000_000
    # The kept tail can overshoot by one 64 KB read: the bound is the cap plus a chunk.
    assert len(res.stdout) < 100_000 + 65_536 + 100 and "bytes not captured" in res.stdout


def test_a_failed_pull_is_a_result_not_an_exception(make_box):
    # Port 1 on this machine refuses at once, so this needs no network.
    res = make_box(image="localhost:1/ct-c10-no-such-image:none").run("true")
    assert not res.ok and "docker pull" in res.command and "refused" in res.stderr


def test_run_tests_says_unavailable_when_the_image_has_no_pytest(make_box, tmp_path):
    ctx = ToolContext(tmp_path, make_box(), FileTracker(), Plan())
    out = run_tests(ctx)
    assert out.is_error and out.data["status"] == "unavailable" and "not installed" in out.content


def test_agent_fixes_the_toy_repo_inside_containers(docker_ready, tmp_path):
    """The whole path: the agent's tests run in containers, its edits reach them through the mount."""
    spec = importlib.util.spec_from_file_location("demo_docker_sandbox", SCRIPTS / "demo_docker_sandbox.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    first, second = demo.fix_bug_in_docker(toy_workspace(tmp_path, git=False))
    assert first.startswith("tests: FAIL") and second.startswith("tests: PASS"), (first, second)
