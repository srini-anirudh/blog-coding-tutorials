"""The permission layer: classify every action by blast radius, ask only when it matters.

Four levels, ordered by how much damage a wrong call can do:

    READ             look at the workspace (read_file, search, git status)
    WORKSPACE_WRITE  change files inside the workspace (write, patch, run tests)
    EXTERNAL         side effects outside the workspace (network, package installs, git push)
    DANGEROUS        hard or impossible to undo (rm -rf outside, sudo, force-push, curl | sh)

A policy auto-approves everything up to a threshold and sends the rest to an
approval callback (a human at a prompt, or a test double).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable


class Risk(IntEnum):
    READ = 0
    WORKSPACE_WRITE = 1
    EXTERNAL = 2
    DANGEROUS = 3


# Patterns are deliberately conservative: a false "dangerous" costs one prompt,
# a false "read" can cost the user's home directory.
_DANGEROUS = [
    r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/|~|\$HOME|\.\.)",  # rm -rf outside the workspace
    r"\bsudo\b", r"\bmkfs\b", r"\bdd\s+if=", r":\(\)\s*\{",      # privilege, disks, fork bomb
    r"\bgit\s+push\b.*(--force|-f\b)", r"\bgit\s+reset\s+--hard\b", r"\bgit\s+clean\s+-[a-z]*f",
    r"(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b", r"\bchmod\s+-R\s+777\b", r">\s*/dev/sd",
]
_EXTERNAL = [
    r"\b(curl|wget|ssh|scp|rsync|nc|telnet)\b", r"\bgit\s+(push|fetch|pull|clone)\b",
    r"\b(pip|pip3|uv|npm|pnpm|yarn|brew|apt|apt-get|cargo)\s+(install|add|remove|uninstall)\b",
    r"\bdocker\b", r"\bkill(all)?\b",
]
_READ_ONLY_PROGRAMS = {
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "pwd", "echo", "which", "file",
    "stat", "diff", "tree", "du", "sort", "uniq", "cut",
}
_READ_ONLY_GIT = {"status", "diff", "log", "show", "blame", "branch", "rev-parse", "ls-files"}


def classify_command(command: str) -> Risk:
    """Risk of a shell command string. Unknown programs default to WORKSPACE_WRITE."""
    for pattern in _DANGEROUS:
        if re.search(pattern, command):
            return Risk.DANGEROUS
    for pattern in _EXTERNAL:
        if re.search(pattern, command):
            return Risk.EXTERNAL
    # Redirection or chaining can turn a read into a write, so only simple commands
    # made entirely of read-only programs count as READ.
    if re.search(r"[>;&`]|\$\(", command):
        return Risk.WORKSPACE_WRITE
    try:
        parts = [shlex.split(seg) for seg in command.split("|")]
    except ValueError:
        return Risk.WORKSPACE_WRITE
    for words in parts:
        if not words:
            return Risk.WORKSPACE_WRITE
        prog = words[0]
        if prog == "git" and len(words) > 1 and words[1] in _READ_ONLY_GIT:
            continue
        if prog == "find" and any(w in ("-delete", "-exec") for w in words):
            return Risk.WORKSPACE_WRITE
        if prog not in _READ_ONLY_PROGRAMS:
            return Risk.WORKSPACE_WRITE
    return Risk.READ


@dataclass
class Decision:
    """The outcome of one permission check. Logged as an Approval event."""

    allowed: bool
    risk: Risk
    reason: str
    asked: bool


# An approver sees the tool name, its arguments and the risk; returns True to allow.
Approver = Callable[[str, dict, Risk], bool]


def deny_all(name: str, arguments: dict, risk: Risk) -> bool:
    return False


@dataclass
class PermissionPolicy:
    """Auto-approve up to `auto_approve`; ask the approver above it.

    `deny_above` is a hard ceiling no approver can lift (e.g. CI runs that must
    never touch the network).
    """

    auto_approve: Risk = Risk.WORKSPACE_WRITE
    approver: Approver = deny_all
    deny_above: Risk = Risk.DANGEROUS

    def check(self, name: str, arguments: dict, risk: Risk) -> Decision:
        if risk > self.deny_above:
            return Decision(False, risk, f"{risk.name} actions are disabled by policy", asked=False)
        if risk <= self.auto_approve:
            return Decision(True, risk, f"auto-approved ({risk.name} <= {self.auto_approve.name})", asked=False)
        allowed = bool(self.approver(name, arguments, risk))
        reason = "approved by user" if allowed else "denied by user"
        return Decision(allowed, risk, reason, asked=True)
