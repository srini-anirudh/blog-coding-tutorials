"""The default coding toolset, and smaller presets for small models.

Every tool schema costs context on every turn. A 0.5B model with twenty tools in
its prompt spends most of its attention on the tool list, so presets matter.
"""

from __future__ import annotations

from . import editing, fs_tools, git_tools, planning, repomap, search, shell, verify
from .tools import ToolRegistry

ALL_TOOLS = [
    fs_tools.read_file, fs_tools.list_dir, fs_tools.write_file,
    editing.edit_file, editing.apply_patch,
    search.search, search.find_files, search.find_symbol, repomap.repo_map,
    shell.run_command,
    verify.run_tests, verify.run_linter, verify.run_typecheck, verify.build,
    planning.update_plan,
    git_tools.git_status, git_tools.git_diff, git_tools.git_log,
    git_tools.checkpoint, git_tools.revert_to_checkpoint,
]

PRESETS = {
    "full": [t.__name__ for t in ALL_TOOLS],
    "minimal": ["read_file", "edit_file", "search", "run_tests"],
    "small-model": ["list_dir", "read_file", "edit_file", "search", "run_tests"],
}


def coding_tools(preset: str = "full") -> ToolRegistry:
    """A fresh registry with the named preset of tools."""
    wanted = set(PRESETS[preset])
    return ToolRegistry([t for t in ALL_TOOLS if t.__name__ in wanted])
