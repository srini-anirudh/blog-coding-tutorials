"""Consolidation: turn many episodes into one reusable rule.

The pattern we look for inside each trajectory is a recovery:

    step i:   action A on entity X fails with error E
    steps ..: the agent runs some fix actions F1..Fk
    step j:   action A on X is retried and succeeds

Abstracting X away gives (A, E, F) templates. A template seen in enough episodes,
across enough different entities, becomes a `Procedure`. Requiring several entities
is what separates a general rule from a quirk of one project.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .episodic import EpisodicStore
from .procedural import Procedure
from .records import Episode

_ERROR_RE = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception))\b")


def error_signature(observation: str) -> str | None:
    """The exception class name, the part of an error that repeats across runs."""
    m = _ERROR_RE.search(observation)
    return m.group(1) if m else None


def templatize(action: str, entity: str) -> str:
    return re.sub(rf"\b{re.escape(entity)}\b", "{entity}", action) if entity else action


@dataclass(frozen=True)
class Recovery:
    trigger: str
    error: str
    fix: tuple[str, ...]
    entity: str
    episode_ids: tuple[str, ...]


def find_recoveries(trajectory: list[Episode]) -> list[Recovery]:
    """Every fail -> fix -> retry-succeeds span in one time-ordered trajectory."""
    found = []
    for i, ep in enumerate(trajectory):
        err = error_signature(ep.observation)
        if ep.reward >= 0 or err is None:
            continue
        for j in range(i + 1, len(trajectory)):
            later = trajectory[j]
            if later.action == ep.action and later.reward > 0:
                # Keep only the fix steps that worked; failed guesses in between are noise.
                fix = tuple(templatize(e.action, ep.entity) for e in trajectory[i + 1:j] if e.reward >= 0)
                if fix:
                    found.append(Recovery(templatize(ep.action, ep.entity), err, fix, ep.entity,
                                          tuple(e.id for e in trajectory[i:j + 1])))
                break
            if later.action == ep.action:
                break  # retried and failed again: this span did not recover
    return found


def consolidate(store: EpisodicStore, min_support: int = 2, min_entities: int = 2) -> list[Procedure]:
    """Group recoveries across all trajectories and promote the recurring ones."""
    groups: dict[tuple[str, str, tuple[str, ...]], list[Recovery]] = defaultdict(list)
    for task in store.tasks():
        for rec in find_recoveries(store.trajectory(task)):
            groups[(rec.trigger, rec.error, rec.fix)].append(rec)
    procs = []
    for (trigger, error, fix), recs in sorted(groups.items()):
        entities = sorted({r.entity for r in recs})
        if len(recs) >= min_support and len(entities) >= min_entities:
            procs.append(Procedure(trigger, error, list(fix), support=len(recs), entities=entities))
    return procs
