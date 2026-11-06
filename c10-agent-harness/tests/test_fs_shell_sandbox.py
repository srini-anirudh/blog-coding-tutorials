"""Steps 4, 5 and 8: filesystem tools, the shell tool, and the sandbox backends."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import pytest
from helpers import make_ctx

from harness.fs_tools import FileTracker, list_dir, read_file, write_file
from harness.sandbox import DockerSandbox, Limits, LocalSandbox, _Capture, network_wrapper
from harness.shell import run_command
from harness.tools import ToolError


def test_read_file_numbers_lines_and_ranges(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    ctx = make_ctx(tmp_path)
    assert "    2 | two" in read_file(ctx, "a.txt")
    out = read_file(ctx, "a.txt", start=2, end=2)
    assert "two" in out and "three" not in out and "[lines 2-2 of 3]" in out


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "a/../../x"])
def test_paths_cannot_escape_the_workspace(tmp_path, path):
    with pytest.raises(ToolError, match="outside the workspace"):
        read_file(make_ctx(tmp_path), path)


def test_symlink_escape_is_blocked(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    (root / "link.txt").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(ToolError, match="outside the workspace"):
        read_file(make_ctx(root), "link.txt")


def test_list_dir_skips_caches(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("")
    out = list_dir(make_ctx(tmp_path))
    assert "src/" in out and "a.py" in out and "__pycache__" not in out


def test_write_requires_a_fresh_read(tmp_path):
    ctx = make_ctx(tmp_path)
    write_file(ctx, "new.txt", "v1\n")              # a new file needs no read
    (tmp_path / "other.txt").write_text("x\n")
    with pytest.raises(ToolError, match="before editing"):
        write_file(ctx, "other.txt", "y\n")
    read_file(ctx, "other.txt")
    (tmp_path / "other.txt").write_text("changed by someone else\n")
    with pytest.raises(ToolError, match="changed since"):
        write_file(ctx, "other.txt", "y\n")


def test_tracker_accepts_own_writes(tmp_path):
    ctx = make_ctx(tmp_path)
    write_file(ctx, "a.txt", "1\n")
    write_file(ctx, "a.txt", "2\n")                 # our own write refreshed the view
    assert (tmp_path / "a.txt").read_text() == "2\n"
    assert isinstance(ctx.files, FileTracker)


def test_shell_reports_streams_and_exit_code(tmp_path):
    out = run_command(make_ctx(tmp_path), "echo out; echo err >&2; exit 3")
    assert out.is_error and out.data["exit_code"] == 3
    assert "exit code: 3" in out.content and "out" in out.content and "--- stderr ---" in out.content


def test_shell_cwd_and_escape(tmp_path):
    (tmp_path / "sub").mkdir()
    ctx = make_ctx(tmp_path)
    assert "sub" in run_command(ctx, "pwd", cwd="sub").content
    with pytest.raises(ToolError, match="escapes"):
        run_command(ctx, "pwd", cwd="../..")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_timeout_kills_the_whole_process_group(tmp_path):
    box = LocalSandbox(tmp_path)
    t0 = time.monotonic()
    res = box.run("sleep 30 & echo $! > child.pid; sleep 30", timeout=1)
    assert res.timed_out and res.exit_code is None and time.monotonic() - t0 < 5
    time.sleep(0.2)
    assert not _alive(int((tmp_path / "child.pid").read_text()))


def test_background_orphans_do_not_hang_the_call(tmp_path):
    box = LocalSandbox(tmp_path)
    t0 = time.monotonic()
    res = box.run("sleep 30 & echo $! > child.pid; echo done", timeout=20)
    assert res.ok and "done" in res.stdout and time.monotonic() - t0 < 5
    time.sleep(0.2)
    assert not _alive(int((tmp_path / "child.pid").read_text()))


def test_capture_is_bounded_and_counts_everything():
    cap = _Capture(1000)
    for _ in range(1000):
        cap.feed(b"x" * 100)
    assert cap.total == 100_000
    assert len(cap.text()) < 1400 and "bytes not captured" in cap.text()


def test_giant_output_stays_in_budget(tmp_path):
    out = run_command(make_ctx(tmp_path), f"{sys.executable} -c \"[print(i) for i in range(50000)]\"")
    assert len(out.content.splitlines()) < 140
    assert "full text saved to .harness/outputs/" in out.content
    assert out.data["exit_code"] == 0


def test_env_is_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    res = LocalSandbox(tmp_path).run("env")
    assert "sk-secret" not in res.stdout and f"HOME={tmp_path}" in res.stdout


def test_from_copy_never_touches_the_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("original")
    box = LocalSandbox.from_copy(src)
    box.run("echo changed > a.txt")
    assert (box.workspace / "a.txt").read_text().strip() == "changed"
    assert (src / "a.txt").read_text() == "original"
    box.cleanup()
    assert not box.workspace.exists()


def test_cpu_limit_kills_a_busy_loop(tmp_path):
    box = LocalSandbox(tmp_path, Limits(cpu_s=1, timeout_s=20))
    res = box.run(f"{sys.executable} -c 'while True: pass'")
    assert not res.ok and not res.timed_out and res.duration_s < 15


def test_file_size_limit(tmp_path):
    box = LocalSandbox(tmp_path, Limits(file_size_mb=1))
    res = box.run(f"{sys.executable} -c \"open('big','wb').write(b'x'*(5*2**20))\"")
    assert res.exit_code != 0 and (tmp_path / "big").stat().st_size <= 2**20


def test_memory_watchdog_kills_the_group(tmp_path):
    box = LocalSandbox(tmp_path, Limits(memory_mb=150, timeout_s=20))
    code = "x=bytearray(800*2**20); import time; x[::4096]=b'1'*len(x[::4096]); time.sleep(30)"
    res = box.run(f"{sys.executable} -c \"{code}\"")
    assert res.killed_reason == "memory" or (res.exit_code not in (0, None))


@pytest.mark.skipif(network_wrapper() is None, reason="this OS offers no per-process network denial")
def test_network_is_denied_by_default(tmp_path):
    code = ("import socket; s=socket.socket(); s.settimeout(2);"
            "\ntry:\n  s.connect(('1.1.1.1',80)); print('connected')\nexcept OSError as e: print('blocked', e)")
    (tmp_path / "net.py").write_text(code)
    assert "blocked" in LocalSandbox(tmp_path).run(f"{sys.executable} net.py").stdout


def test_docker_argv_carries_every_limit(tmp_path):
    box = DockerSandbox(tmp_path, limits=Limits(memory_mb=512, cpu_s=30, file_size_mb=10), cpus=1.5)
    argv = box.docker_argv("pytest -q", cwd="src", name="t")
    joined = " ".join(argv)
    assert "--network none" in joined and "--cpus 1.5" in joined and "--memory 512m" in joined
    assert "--cap-drop ALL" in joined and f"{tmp_path.resolve()}:/workspace:rw" in joined
    assert "-w /workspace/src" in joined and argv[-3:] == ["sh", "-c", "pytest -q"]


def test_docker_network_flag_opens_bridge(tmp_path):
    argv = DockerSandbox(tmp_path, limits=Limits(network=True)).docker_argv("true")
    assert argv[argv.index("--network") + 1] == "bridge"


def test_docker_runs_for_real(docker_ready, tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    assert "hello" in DockerSandbox(tmp_path).run("cat a.txt", timeout=120).stdout


def test_path_helper_is_used_by_sandbox(tmp_path):
    assert LocalSandbox(tmp_path).resolve_cwd(".") == Path(tmp_path).resolve()


def test_footer_and_spill_tell_the_truth_when_capture_dropped_bytes(tmp_path):
    """Past 1 MB, `_Capture` loses the middle before truncation sees it. Nothing may claim "full text"."""
    (tmp_path / "gen.py").write_text("for i in range(20000):\n"
                                     "    print('E   AssertionError: lost' if i == 10000 else f'{i:<99}')\n")
    out = run_command(make_ctx(tmp_path), f"{sys.executable} gen.py")
    footer = next(ln for ln in out.content.splitlines() if ln.startswith("[output truncated"))
    assert "full text" not in footer and "AssertionError" not in out.content
    dropped = int(re.search(r"; (\d+) bytes were dropped", footer)[1])

    spill = next((tmp_path / ".harness" / "outputs").glob("*-stdout.txt")).read_text()
    header, body = spill.split("\n", 1)
    assert f"{dropped} bytes" in header and "AssertionError" not in spill
    marker = f"\n... [{dropped} bytes not captured] ...\n"
    stdout_bytes = int(re.search(r"(\d+) B stdout", out.content)[1])
    assert stdout_bytes == dropped + len(body.replace(marker, "", 1).encode())  # the number is exact


def test_footer_still_promises_the_full_text_when_nothing_was_dropped(tmp_path):
    (tmp_path / "gen.py").write_text("for i in range(5000):\n    print(i)\n")
    out = run_command(make_ctx(tmp_path), f"{sys.executable} gen.py")
    assert "5000 lines total; full text saved to" in out.content and "dropped" not in out.content
    spill = next((tmp_path / ".harness" / "outputs").glob("*-stdout.txt")).read_text()
    assert spill.splitlines()[0] == "0" and len(spill.splitlines()) == 5000
