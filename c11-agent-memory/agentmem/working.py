"""Working memory: the explicit task state the agent carries from step to step.

Working memory is not retrieved. It is always in context, it is small, and the agent
rewrites it as the task moves. Keeping it as a structured object (instead of hoping the
model re-derives it from the transcript) is what lets compaction throw the transcript
away without losing the plot.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PlanStep:
    text: str
    done: bool = False


@dataclass
class TaskState:
    """The goal, where we are, and what is still unknown."""

    goal: str
    current_step: str = ""
    plan: list[PlanStep] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def apply(self, update: dict[str, Any]) -> None:
        """Merge an update the model emitted, e.g. {"done": ["inspect"], "current_step": "fix"}.

        Updates are small deltas, not full rewrites, so a malformed update can only
        damage the fields it names.
        """
        if "current_step" in update:
            self.current_step = str(update["current_step"])
        for text in update.get("add_steps", []):
            self.plan.append(PlanStep(str(text)))
        for text in update.get("done", []):
            for step in self.plan:
                if step.text == text:
                    step.done = True
        for q in update.get("ask", []):
            if q not in self.open_questions:
                self.open_questions.append(str(q))
        for q in update.get("answered", []):
            if q in self.open_questions:
                self.open_questions.remove(q)
        for c in update.get("constraints", []):
            if c not in self.constraints:
                self.constraints.append(str(c))

    def render(self) -> str:
        """The block that goes into context. Checkboxes read well to models and humans."""
        lines = [f"goal: {self.goal}"]
        if self.current_step:
            lines.append(f"current step: {self.current_step}")
        lines += [f"[{'x' if s.done else ' '}] {s.text}" for s in self.plan]
        lines += [f"open question: {q}" for q in self.open_questions]
        lines += [f"constraint: {c}" for c in self.constraints]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "TaskState":
        d = json.loads(s)
        d["plan"] = [PlanStep(**p) for p in d.get("plan", [])]
        return cls(**d)
