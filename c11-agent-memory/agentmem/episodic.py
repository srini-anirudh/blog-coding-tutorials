"""Episodic memory: an append-only log of (s, a, o, r, t) transitions.

Episodes are the raw material. They are cheap to write (no judgement needed) and too
verbose to put in context directly, so the rest of the package reads from them:
retrieval returns a few, consolidation turns many into one rule.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .records import Episode


class EpisodicStore:
    def __init__(self) -> None:
        self.episodes: list[Episode] = []

    def add(self, episode: Episode) -> Episode:
        self.episodes.append(episode)
        return episode

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self) -> Iterator[Episode]:
        return iter(self.episodes)

    def trajectory(self, task: str, session: int | None = None) -> list[Episode]:
        """All steps of one task in time order: the unit consolidation works on."""
        steps = [e for e in self.episodes if e.task == task and (session is None or e.session == session)]
        return sorted(steps, key=lambda e: e.t)

    def tasks(self) -> list[str]:
        return sorted({e.task for e in self.episodes})

    def recent(self, n: int) -> list[Episode]:
        return sorted(self.episodes, key=lambda e: e.t)[-n:]

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for e in self.episodes:
                f.write(json.dumps(asdict(e), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "EpisodicStore":
        store = cls()
        with open(path) as f:
            for line in f:
                if line.strip():
                    store.add(Episode(**json.loads(line)))
        return store
