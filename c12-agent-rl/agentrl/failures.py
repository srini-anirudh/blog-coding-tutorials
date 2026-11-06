"""Failure injection: the ways real tools go wrong, made reproducible.

Real agents meet tools that hang, crash, reject arguments, and files that moved. A policy trained only on clean
runs has never seen an error message, so it has no idea what to do next. These failures are injected by the
environment, seeded per episode, so that a recovery behaviour can be measured and rewarded.

    timeout        the tool hangs; the call returns "error timeout" and costs a step   (search, read)
    tool error     the tool crashes; the call returns "error internal"                  (any tool)
    invalid input  the harness garbles an argument; the call returns "error invalid input"
    missing file   the task names a stale path; produced by the task generator (kind "stale")

All of them are transient except the stale path: retrying the same call can succeed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

TIMEOUT = "error timeout"
TOOL_ERROR = "error internal"
INVALID = "error invalid input"
MISSING = "error no such file"
TRANSIENT = (TIMEOUT, TOOL_ERROR, INVALID)


@dataclass(frozen=True)
class FailureConfig:
    p_timeout: float = 0.0
    p_tool_error: float = 0.0
    p_invalid: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.p_timeout + self.p_tool_error + self.p_invalid > 0


def inject(cfg: FailureConfig, command: str, rng: random.Random) -> str | None:
    """Return an error message to replace the tool's real output, or None to let the call through."""
    if command == "submit":
        return None
    if command in ("search", "read") and rng.random() < cfg.p_timeout:
        return TIMEOUT
    if rng.random() < cfg.p_tool_error:
        return TOOL_ERROR
    if command == "edit" and rng.random() < cfg.p_invalid:
        return INVALID
    return None
