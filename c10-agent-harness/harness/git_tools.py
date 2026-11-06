"""Git integration: status, diff, log, and checkpoints that never touch the user's branch.

A checkpoint is a commit object built from a temporary index and stored under
refs/harness/checkpoints/. The user's HEAD, branch and staging area are left alone,
so the agent can save and restore the working tree as often as it likes.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .permissions import Risk
from .tools import ToolContext, ToolError, tool

REF_PREFIX = "refs/harness/checkpoints/"


def git(root: Path, *args: str, env: dict | None = None, check: bool = True) -> str:
    res = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                         env={**os.environ, **(env or {})})
    if check and res.returncode != 0:
        raise ToolError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def is_repo(root: Path) -> bool:
    return (Path(root) / ".git").exists()


def init_repo(root: Path, message: str = "initial commit") -> None:
    """Make `root` a git repo with one commit (used for temp workspaces)."""
    ident = {"GIT_AUTHOR_NAME": "harness", "GIT_AUTHOR_EMAIL": "harness@example.com",
             "GIT_COMMITTER_NAME": "harness", "GIT_COMMITTER_EMAIL": "harness@example.com"}
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message, env=ident)


def create_checkpoint(root: Path, label: str) -> str:
    """Snapshot the working tree (tracked + untracked, minus ignored). Returns the commit sha."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index"),
               "GIT_AUTHOR_NAME": "harness", "GIT_AUTHOR_EMAIL": "harness@example.com",
               "GIT_COMMITTER_NAME": "harness", "GIT_COMMITTER_EMAIL": "harness@example.com"}
        git(root, "add", "-A", env=env)
        tree = git(root, "write-tree", env=env).strip()
    head = git(root, "rev-parse", "--verify", "-q", "HEAD", check=False).strip()
    parent = ["-p", head] if head else []
    sha = git(root, "commit-tree", tree, *parent, "-m", f"harness checkpoint: {label}", env=env).strip()
    n = len(list_checkpoints(root))
    git(root, "update-ref", f"{REF_PREFIX}{n:03d}", sha)
    return sha


def list_checkpoints(root: Path) -> list[tuple[str, str]]:
    out = git(root, "for-each-ref", "--format=%(refname) %(objectname)", REF_PREFIX, check=False)
    return [tuple(line.split()) for line in out.splitlines() if line.strip()]  # type: ignore[misc]


def restore_checkpoint(root: Path, sha: str) -> list[str]:
    """Make the working tree equal the checkpoint. Returns paths that were deleted."""
    git(root, "cat-file", "-e", f"{sha}^{{commit}}")
    in_checkpoint = set(git(root, "ls-tree", "-r", "--name-only", sha).splitlines())
    present = set(git(root, "ls-files", "--cached", "--others", "--exclude-standard").splitlines())
    removed = sorted(p for p in present - in_checkpoint if (Path(root) / p).exists())
    for p in removed:
        (Path(root) / p).unlink()
    git(root, "restore", f"--source={sha}", "--worktree", "--", ".")
    return removed


@tool(risk=Risk.READ)
def git_status(ctx: ToolContext) -> str:
    """Show changed, staged and untracked files (git status --short)."""
    return git(Path(ctx.workspace), "status", "--short", "--branch").strip() or "clean"


@tool(risk=Risk.READ)
def git_diff(ctx: ToolContext, path: str = "", staged: bool = False) -> str:
    """Show the diff of uncommitted changes.

    Args:
        path: Limit the diff to one file or directory.
        staged: Show staged changes instead of working-tree changes.
    """
    args = ["diff", "--no-color"] + (["--staged"] if staged else []) + (["--", path] if path else [])
    return git(Path(ctx.workspace), *args).strip() or "no changes"


@tool(risk=Risk.READ)
def git_log(ctx: ToolContext, n: int = 10) -> str:
    """Show recent commits, one per line.

    Args:
        n: Number of commits.
    """
    return git(Path(ctx.workspace), "log", f"-{n}", "--oneline", "--no-color").strip()


@tool(risk=Risk.WORKSPACE_WRITE)
def checkpoint(ctx: ToolContext, label: str) -> str:
    """Save the current working tree so it can be restored later with revert_to_checkpoint.

    Args:
        label: A short description, for example "before refactor".
    """
    sha = create_checkpoint(Path(ctx.workspace), label)
    return f"checkpoint {sha[:10]} saved ({label})"


@tool(risk=Risk.WORKSPACE_WRITE)
def revert_to_checkpoint(ctx: ToolContext, checkpoint_id: str) -> str:
    """Restore the working tree to a checkpoint, discarding later edits.

    Args:
        checkpoint_id: The id printed by checkpoint (a commit sha prefix).
    """
    removed = restore_checkpoint(Path(ctx.workspace), checkpoint_id)
    extra = f"; removed {', '.join(removed)}" if removed else ""
    return f"restored working tree to {checkpoint_id[:10]}{extra}"
