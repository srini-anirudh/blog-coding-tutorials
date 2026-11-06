"""Planning state: a checklist the model maintains and the harness re-shows every turn.

    [x] inspect the repository
    [x] reproduce the failing test
    [>] fix median for even-length input
    [ ] run the tests again

The plan lives outside the chat history, so it survives compaction. That is the
point: after the history is summarised, the model still knows where it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .tools import ToolContext, ToolError, tool

MARKS = {"[ ]": "pending", "[>]": "in_progress", "[x]": "done", "[-]": "skipped"}
SYMBOL = {v: k for k, v in MARKS.items()}


@dataclass
class PlanStep:
    title: str
    status: str = "pending"


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)

    def render(self) -> str:
        return "\n".join(f"{SYMBOL[s.status]} {s.title}" for s in self.steps) or "(no plan yet)"

    @property
    def current(self) -> PlanStep | None:
        return next((s for s in self.steps if s.status == "in_progress"), None)

    def set_from_lines(self, lines: list[str]) -> None:
        steps = []
        for raw in lines:
            m = re.match(r"^\s*(\[[ >x\-]\])\s*(.+)$", raw)
            if not m:
                raise ToolError(f"plan item {raw!r} must start with [ ], [>], [x] or [-]")
            steps.append(PlanStep(m.group(2).strip(), MARKS[m.group(1)]))
        if sum(s.status == "in_progress" for s in steps) > 1:
            raise ToolError("at most one plan item can be in progress ([>])")
        self.steps = steps


@tool
def update_plan(ctx: ToolContext, items: list[str]) -> str:
    """Replace the task plan. Mark items [ ] pending, [>] in progress, [x] done, [-] skipped.

    Args:
        items: The full plan, one item per entry, for example ["[x] reproduce", "[>] fix", "[ ] test"].
    """
    if ctx.plan is None:
        raise ToolError("planning is not enabled")
    ctx.plan.set_from_lines(items)
    return "plan updated:\n" + ctx.plan.render()
