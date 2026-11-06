"""The context budgeter: given a token budget B, decide what the model sees.

Candidates are `Block`s, each tagged with a section, a token cost and a value. The
packer is greedy by value density (value per token), the standard relaxation of the
knapsack problem, with three additions that matter in practice:

1. required blocks (instructions, task state, the request) go in first, or packing fails loudly;
2. each section can reserve a floor and be held under a cap, so one giant tool output
   cannot starve the memories, and the memories cannot starve recent events;
3. a block that does not fit may be compressed into the room that is left instead of dropped.

The output is rendered in a fixed section order (stable text first, volatile text last),
not in score order: the order of a prompt changes how it is used and how well it caches.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from .tokens import count_tokens, truncate_to_tokens


@dataclass
class Block:
    section: str
    text: str
    value: float = 0.0          # usefulness estimate; required blocks ignore it
    order: float = 0.0          # position inside its section when rendered
    required: bool = False
    compressible: bool = True
    id: str = ""
    overhead: int = 0           # tokens the block costs beyond its text, e.g. chat framing


@dataclass
class Section:
    name: str
    title: str
    min_tokens: int = 0              # reserved before the global greedy pass
    max_tokens: int | None = None    # cap for the whole section


DEFAULT_SECTIONS = [
    Section("instructions", "Instructions"),
    Section("task", "Task state"),
    Section("memory", "Relevant memory", max_tokens=400),
    Section("events", "Recent events", min_tokens=60),
    Section("observation", "Latest observation", max_tokens=300),
    Section("request", "Request"),
]


class BudgetError(ValueError):
    pass


@dataclass
class Packed:
    budget: int
    blocks: list[Block]
    dropped: list[Block]
    compressed: list[str] = field(default_factory=list)   # ids of blocks that were shortened
    tokens_by_section: dict[str, int] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.tokens_by_section.values())

    def render(self) -> str:
        parts = []
        for sec in self.sections:
            body = [b.text for b in sorted((b for b in self.blocks if b.section == sec.name), key=lambda b: b.order)]
            if body:
                parts.append(f"## {sec.title}\n" + "\n".join(body))
        return "\n\n".join(parts)


Compressor = Callable[[Block, int], str]


def default_compressor(block: Block, max_tokens: int) -> str:
    """Tool output keeps its head and tail; everything else keeps its beginning."""
    keep = "both" if block.section == "observation" else "head"
    return truncate_to_tokens(block.text, max_tokens, keep=keep)


class ContextBudgeter:
    def __init__(self, sections: list[Section] | None = None, counter: Callable[[str], int] = count_tokens,
                 compressor: Compressor = default_compressor, min_compressed: int = 16) -> None:
        self.sections = sections or DEFAULT_SECTIONS
        self.by_name = {s.name: s for s in self.sections}
        self.counter = counter
        self.compressor = compressor
        self.min_compressed = min_compressed   # a fragment shorter than this is not worth its header

    def cost(self, block: Block) -> int:
        return self.counter(block.text) + 1 + block.overhead  # +1 for the joining newline

    def header_cost(self, name: str) -> int:
        return self.counter(f"## {self.by_name[name].title}") + 2

    def pack(self, blocks: list[Block], budget: int) -> Packed:
        used: dict[str, int] = {s.name: 0 for s in self.sections}
        chosen: list[Block] = []      # what goes in (compressed copies where shortened)
        taken: set[int] = set()       # id() of the original blocks already placed
        compressed: list[str] = []

        def room(sec: str, limit: int | None) -> int:
            header = 0 if used[sec] else self.header_cost(sec)
            space = budget - sum(used.values()) - header
            for cap in (self.by_name[sec].max_tokens, limit):
                if cap is not None:
                    space = min(space, cap - used[sec] - header)
            return space

        def try_add(b: Block, limit: int | None = None) -> bool:
            space = room(b.section, limit)
            if self.cost(b) <= space:
                placed = b
            elif b.compressible and space - 1 - b.overhead >= self.min_compressed:
                text = self.compressor(b, space - 1 - b.overhead)
                if not text or self.counter(text) + 1 + b.overhead > space:
                    return False
                placed = replace(b, text=text)
                compressed.append(b.id)
            else:
                return False
            used[b.section] += self.cost(placed) + (0 if used[b.section] else self.header_cost(b.section))
            chosen.append(placed)
            taken.add(id(b))
            return True

        # 1. Required blocks, in section order. If they cannot fit, no packing is valid.
        for b in sorted((b for b in blocks if b.required), key=lambda b: self._rank(b.section)):
            if not try_add(b):
                required = [x for x in blocks if x.required]
                need = sum(map(self.cost, required)) + sum(self.header_cost(n) for n in {x.section for x in required})
                raise BudgetError(f"required context needs ~{need} tokens, budget is {budget}")

        optional = sorted((b for b in blocks if not b.required), key=self._density, reverse=True)
        # 2. Section floors: each section first gets its reserved tokens from its own best blocks.
        for sec in self.sections:
            if sec.min_tokens:
                for b in optional:
                    if b.section == sec.name and id(b) not in taken:
                        try_add(b, limit=sec.min_tokens)
        # 3. Global greedy by value per token across everything left.
        for b in optional:
            if id(b) not in taken:
                try_add(b)

        dropped = [b for b in blocks if id(b) not in taken]
        return Packed(budget, chosen, dropped, compressed, {k: v for k, v in used.items() if v}, self.sections)

    def _density(self, b: Block) -> float:
        return b.value / max(1, self.cost(b))

    def _rank(self, section: str) -> int:
        return [s.name for s in self.sections].index(section)


def truncate_left(blocks: list[Block], budget: int, counter: Callable[[str], int] = count_tokens) -> list[Block]:
    """The naive baseline: concatenate in time order and keep the newest B tokens.

    This is what "just send the transcript" does once the transcript outgrows the
    window. The oldest text goes first, whatever it was, instructions included.
    """
    kept: list[Block] = []
    used = 0
    for b in sorted(blocks, key=lambda b: b.order, reverse=True):
        c = counter(b.text) + 1
        if used + c > budget:
            break
        kept.append(b)
        used += c
    return kept[::-1]
