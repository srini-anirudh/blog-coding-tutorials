"""Where commands actually run.

Two backends with one interface (`Sandbox.run(command, timeout) -> ExecResult`):

* `LocalSandbox`: a temp copy of the workspace, a scrubbed environment, rlimits
  (CPU seconds, file size, open files), a memory watchdog, and network denial via
  macOS `sandbox-exec` or Linux `unshare -n` when available. Good for laptops and CI.
* `DockerSandbox`: the same contract inside a container with `--network none`,
  `--cpus`, `--memory`, `--pids-limit` and a bind-mounted workspace. Stronger
  isolation, needs a Docker daemon.

Both share `run_process`, which owns the hard part: timeouts that kill the whole
process group, and output capture that cannot run out of memory.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


@dataclass
class ExecResult:
    command: str
    exit_code: int | None           # None if we killed it before it exited
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    killed_reason: str | None = None  # "timeout" | "memory" | None
    stdout_bytes: int = 0             # total produced, even if capture dropped some
    stderr_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class Limits:
    timeout_s: float = 60.0
    cpu_s: int | None = 120           # RLIMIT_CPU: kernel sends SIGXCPU
    memory_mb: int | None = 2048      # watchdog everywhere, RLIMIT_AS on Linux too
    file_size_mb: int | None = 256    # RLIMIT_FSIZE: a runaway log cannot fill the disk
    open_files: int | None = 1024
    network: bool = False             # False = deny where the OS lets us
    capture_bytes: int = 1_000_000    # per stream, head + tail kept


class Sandbox(Protocol):
    workspace: Path
    python: str

    def run(self, command: str, timeout: float | None = None, cwd: str = ".") -> ExecResult: ...


# ---------------------------------------------------------------- process core

class _Capture:
    """Keep the first and last `cap/2` bytes of a stream. Constant memory for any output size."""

    def __init__(self, cap: int):
        self.half = cap // 2
        self.head = bytearray()
        self.tail: deque[bytes] = deque()
        self.tail_len = 0
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self.half - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self.tail.append(chunk)
            self.tail_len += len(chunk)
            while self.tail_len - len(self.tail[0]) >= self.half:
                self.tail_len -= len(self.tail.popleft())

    def text(self) -> str:
        tail = b"".join(self.tail)
        dropped = self.total - len(self.head) - len(tail)
        mid = f"\n... [{dropped} bytes not captured] ...\n".encode() if dropped > 0 else b""
        return (bytes(self.head) + mid + tail).decode("utf-8", errors="replace")


def _pump(stream, cap: _Capture) -> None:
    for chunk in iter(lambda: stream.read1(65536), b""):
        cap.feed(chunk)


def _group_rss_mb(pgid: int) -> float:
    """Resident memory of every process in a group (ps is portable across macOS and Linux)."""
    out = subprocess.run(["ps", "-axo", "pgid=,rss="], capture_output=True, text=True).stdout
    kb = sum(int(r) for g, r in (ln.split() for ln in out.splitlines() if ln.strip()) if int(g) == pgid)
    return kb / 1024


def kill_group(pgid: int, grace_s: float = 0.5) -> None:
    """SIGTERM the group, then SIGKILL whatever is left. Orphans included."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(grace_s if sig == signal.SIGTERM else 0)


def run_process(argv: list[str], cwd: Path, env: dict[str, str], timeout_s: float,
                preexec: Callable[[], None] | None = None, memory_mb: int | None = None,
                capture_bytes: int = 1_000_000, label: str | None = None,
                on_kill: Callable[[], None] | None = None) -> ExecResult:
    """Run argv in its own session; enforce the timeout and memory cap on the whole group."""
    start = time.monotonic()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True, preexec_fn=preexec)
    pgid = proc.pid  # start_new_session makes the child a group leader
    out, err = _Capture(capture_bytes), _Capture(capture_bytes)
    readers = [threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True),
               threading.Thread(target=_pump, args=(proc.stderr, err), daemon=True)]
    for t in readers:
        t.start()

    killed: str | None = None
    ticks = 0
    while proc.poll() is None:
        ticks += 1
        if time.monotonic() - start > timeout_s:
            killed = "timeout"
        elif memory_mb is not None and ticks % 5 == 0 and _group_rss_mb(pgid) > memory_mb:
            killed = "memory"  # polled every ~0.25 s: `ps` is too slow to call every tick
        if killed:
            if on_kill:
                on_kill()
            kill_group(pgid)
            break
        try:
            proc.wait(timeout=0.05)  # returns as soon as the child exits; a sleep would add 50 ms to every command
        except subprocess.TimeoutExpired:
            pass
    proc.wait()
    # The leader exited, but `cmd &` children may still hold the pipes open.
    # Kill the rest of the group so nothing outlives the tool call.
    kill_group(pgid, grace_s=0.1)
    for t in readers:
        t.join(timeout=2)
    return ExecResult(command=label or " ".join(argv),
                      exit_code=None if killed else proc.returncode,
                      stdout=out.text(), stderr=err.text(),
                      duration_s=time.monotonic() - start,
                      timed_out=killed == "timeout", killed_reason=killed,
                      stdout_bytes=out.total, stderr_bytes=err.total)


# ---------------------------------------------------------------- local backend

_SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SYSTEMROOT")
_COPY_IGNORE = shutil.ignore_patterns(".venv", "__pycache__", ".mypy_cache", ".pytest_cache",
                                      "node_modules", "*.pyc")
_DENY_NETWORK_PROFILE = "(version 1)(allow default)(deny network*)"


def _rlimits(limits: Limits) -> Callable[[], None]:
    import resource

    def apply() -> None:
        def cap(res: int, value: int) -> None:
            try:
                resource.setrlimit(res, (value, value))
            except (ValueError, OSError):
                pass  # some kernels refuse some limits; the watchdog still covers memory
        if limits.cpu_s:
            cap(resource.RLIMIT_CPU, limits.cpu_s)
        if limits.file_size_mb:
            cap(resource.RLIMIT_FSIZE, limits.file_size_mb * 2**20)
        if limits.open_files:
            cap(resource.RLIMIT_NOFILE, limits.open_files)
        if limits.memory_mb and sys.platform.startswith("linux"):
            cap(resource.RLIMIT_AS, limits.memory_mb * 2**20)  # macOS does not enforce RLIMIT_AS
    return apply


def network_wrapper() -> list[str] | None:
    """argv prefix that denies network for the child, or None if this OS offers none."""
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists():
        return ["/usr/bin/sandbox-exec", "-p", _DENY_NETWORK_PROFILE]
    if sys.platform.startswith("linux") and shutil.which("unshare"):
        return ["unshare", "--net", "--map-root-user"]
    return None


@dataclass
class LocalSandbox:
    """Restricted subprocess sandbox rooted at `workspace`.

    Not a security boundary against a determined attacker (same user, same kernel).
    It is a guard rail against accidents: runaway loops, giant outputs, stray network calls.
    """

    workspace: Path
    limits: Limits = field(default_factory=Limits)
    extra_env: dict[str, str] = field(default_factory=dict)
    owns_workspace: bool = False
    python: str = sys.executable      # interpreter used by the verification tools

    @classmethod
    def from_copy(cls, source: Path, limits: Limits | None = None, **kw) -> "LocalSandbox":
        """Copy `source` into a fresh temp directory and work there; the original is never touched."""
        root = Path(tempfile.mkdtemp(prefix="harness-ws-"))
        dest = root / Path(source).name
        shutil.copytree(source, dest, ignore=_COPY_IGNORE, symlinks=True)
        return cls(dest, limits or Limits(), owns_workspace=True, **kw)

    def env(self) -> dict[str, str]:
        env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
        # No API keys, no tokens, no user config leak into commands the model writes.
        env.update(HOME=str(self.workspace), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        if not self.limits.network:
            env.update(HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", NO_PROXY="")
        env.update(self.extra_env)
        return env

    def resolve_cwd(self, cwd: str) -> Path:
        path = (self.workspace / cwd).resolve()
        if not path.is_relative_to(self.workspace.resolve()):
            raise ValueError(f"cwd {cwd!r} escapes the workspace")
        return path

    def run(self, command: str, timeout: float | None = None, cwd: str = ".") -> ExecResult:
        argv = ["/bin/sh", "-c", command]
        if not self.limits.network and (wrap := network_wrapper()):
            argv = wrap + argv
        return run_process(argv, self.resolve_cwd(cwd), self.env(),
                           timeout_s=timeout or self.limits.timeout_s,
                           preexec=_rlimits(self.limits), memory_mb=self.limits.memory_mb,
                           capture_bytes=self.limits.capture_bytes, label=command)

    def cleanup(self) -> None:
        if self.owns_workspace:
            shutil.rmtree(self.workspace.parent, ignore_errors=True)


# ---------------------------------------------------------------- docker backend

def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@dataclass
class DockerSandbox:
    """Run each command in a fresh container with the workspace bind-mounted at /workspace."""

    workspace: Path
    image: str = "python:3.12-slim"
    limits: Limits = field(default_factory=Limits)
    cpus: float = 2.0
    pids_limit: int = 256
    python: str = "python"

    def docker_argv(self, command: str, cwd: str = ".", name: str | None = None) -> list[str]:
        lim = self.limits
        argv = ["docker", "run", "--rm", "-i",
                "--name", name or f"harness-{uuid.uuid4().hex[:8]}",
                "--network", "bridge" if lim.network else "none",
                "--cpus", str(self.cpus), "--pids-limit", str(self.pids_limit),
                "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
                "-v", f"{Path(self.workspace).resolve()}:/workspace:rw",
                "-w", f"/workspace/{cwd}".rstrip("/.") or "/workspace",
                "-e", "HOME=/workspace", "-e", "PYTHONDONTWRITEBYTECODE=1"]
        if lim.memory_mb:
            argv += ["--memory", f"{lim.memory_mb}m", "--memory-swap", f"{lim.memory_mb}m"]
        if lim.file_size_mb:
            argv += ["--ulimit", f"fsize={lim.file_size_mb * 2**20}"]
        if lim.cpu_s:
            argv += ["--ulimit", f"cpu={lim.cpu_s}"]
        return argv + [self.image, "sh", "-c", command]

    def run(self, command: str, timeout: float | None = None, cwd: str = ".") -> ExecResult:
        name = f"harness-{uuid.uuid4().hex[:8]}"
        # Killing the docker CLI does not stop the container, so the kill hook removes it.
        def stop() -> None:
            subprocess.run(["docker", "kill", name], capture_output=True)
        return run_process(self.docker_argv(command, cwd, name), Path(self.workspace),
                           dict(os.environ), timeout_s=timeout or self.limits.timeout_s,
                           memory_mb=None,  # the cgroup enforces memory inside docker
                           capture_bytes=self.limits.capture_bytes, label=command, on_kill=stop)
