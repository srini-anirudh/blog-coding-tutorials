"""Shared test fixtures: a throwaway git repo seeded from the toy project."""

from __future__ import annotations

import shutil
from pathlib import Path

from harness import LocalSandbox
from harness.fs_tools import FileTracker
from harness.git_tools import init_repo
from harness.planning import Plan
from harness.tools import ToolContext

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy_repo"


def toy_workspace(tmp_path: Path, git: bool = True) -> Path:
    dest = tmp_path / "toy"
    shutil.copytree(TOY, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    if git:
        init_repo(dest)
    return dest


def make_ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace, LocalSandbox(workspace), FileTracker(), Plan())
