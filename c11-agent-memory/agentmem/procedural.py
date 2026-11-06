"""Procedural memory: reusable strategies, stored as templated steps with a track record.

A procedure says what to do, not what is true. It carries its own evidence (how many
episodes it came from, how often it worked since), so a strategy that stops working
loses rank instead of being followed forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .records import new_id


@dataclass
class Procedure:
    """"When `trigger` fails with `error`, run `steps`, then retry."

    Templates use `{entity}` for the object, so one procedure covers every project.
    """

    trigger: str              # e.g. "deploy {entity}"
    error: str                # e.g. "MigrationError"
    steps: list[str]          # e.g. ["migrate {entity}"]
    support: int = 1          # episodes it was consolidated from
    entities: list[str] = field(default_factory=list)
    successes: int = 0        # outcomes recorded after it was written
    failures: int = 0
    id: str = field(default_factory=lambda: new_id("p"))

    @property
    def success_rate(self) -> float:
        """Laplace-smoothed, so a brand-new procedure starts at 0.5, not at 0 or 1."""
        return (self.successes + 1) / (self.successes + self.failures + 2)

    def instantiate(self, entity: str) -> list[str]:
        return [s.format(entity=entity) for s in self.steps]

    def text(self) -> str:
        steps = ", then ".join(f"`{s}`" for s in self.steps)
        return (f"rule: when `{self.trigger}` fails with {self.error}, run {steps} and retry "
                f"(from {self.support} episodes over {len(self.entities)} projects)")


class ProcedureStore:
    def __init__(self) -> None:
        self.procedures: list[Procedure] = []

    def add(self, proc: Procedure) -> Procedure:
        """Merge with an existing procedure that has the same trigger, error and steps."""
        for p in self.procedures:
            if (p.trigger, p.error, p.steps) == (proc.trigger, proc.error, proc.steps):
                p.support = max(p.support, proc.support)
                p.entities = sorted(set(p.entities) | set(proc.entities))
                return p
        self.procedures.append(proc)
        return proc

    def lookup(self, trigger: str, error: str) -> Procedure | None:
        """The best procedure for this failure, ranked by its track record."""
        matches = [p for p in self.procedures if p.trigger == trigger and p.error == error]
        return max(matches, key=lambda p: (p.success_rate, p.support), default=None)

    def record_outcome(self, proc_id: str, success: bool) -> None:
        for p in self.procedures:
            if p.id == proc_id:
                if success:
                    p.successes += 1
                else:
                    p.failures += 1

    def __len__(self) -> int:
        return len(self.procedures)
