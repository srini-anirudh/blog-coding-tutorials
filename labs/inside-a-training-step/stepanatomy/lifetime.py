"""Watch activation memory rise through the forward pass and drain through the backward.

Every tensor autograd saves for backward passes through a pack hook. We wrap it in a small holder
object and attach a finalizer to the holder. Autograd drops the holder as soon as the node that
needed it has run its backward, so the finalizer fires at the exact moment the saved tensor dies.
That gives a timeline of live saved bytes with no allocator involved, so it runs on CPU.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field

import torch
from torch import nn


class _Held:
    __slots__ = ("t", "__weakref__")

    def __init__(self, t: torch.Tensor) -> None:
        self.t = t


@dataclass
class Event:
    phase: str  # "forward" or "backward"
    tag: str  # which block (or "head") the saved tensor came from
    live_bytes: int


@dataclass
class Timeline:
    events: list[Event] = field(default_factory=list)
    turn: int = 0  # index of the last forward event: the instant the loss exists
    # tag -> [index of its first save, index of its last release]
    spans: dict[str, list[int]] = field(default_factory=dict)
    saved_by_tag: dict[str, int] = field(default_factory=dict)
    shapes_by_tag: dict[str, list] = field(default_factory=dict)  # tag -> [(shape, dtype, storage bytes)]

    @property
    def peak(self) -> int:
        return max(e.live_bytes for e in self.events)

    @property
    def peak_index(self) -> int:
        return max(range(len(self.events)), key=lambda i: self.events[i].live_bytes)


class SavedTensorTracker:
    """Track tensors saved for backward, excluding parameters (they are not activations).

    Storages are reference counted, because several ops can save the same tensor.
    """

    def __init__(self, model: nn.Module, blocks: list[nn.Module]) -> None:
        self.param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
        self.blocks = blocks
        self.tl = Timeline()
        self.phase = "forward"
        self.tag = "embed"
        self.refs: dict[int, int] = {}
        self.size: dict[int, int] = {}
        self.owner: dict[int, str] = {}
        self.live = 0

    def _record(self, tag: str) -> None:
        self.tl.events.append(Event(self.phase, tag, self.live))
        span = self.tl.spans.setdefault(tag, [len(self.tl.events) - 1, len(self.tl.events) - 1])
        span[1] = len(self.tl.events) - 1

    def _pack(self, t: torch.Tensor):
        st = t.untyped_storage()
        key = st.data_ptr()
        if key in self.param_ptrs or st.nbytes() == 0:
            return t
        if self.refs.get(key, 0) == 0:
            self.size[key] = st.nbytes()
            self.owner[key] = self.tag
            self.live += st.nbytes()
            self.tl.saved_by_tag[self.tag] = self.tl.saved_by_tag.get(self.tag, 0) + st.nbytes()
            self.tl.shapes_by_tag.setdefault(self.tag, []).append((tuple(t.shape), str(t.dtype), st.nbytes()))
            self.refs[key] = 0
            self._record(self.tag)
        self.refs[key] += 1
        held = _Held(t)
        weakref.finalize(held, self._release, key)
        return held

    def _release(self, key: int) -> None:
        self.refs[key] -= 1
        if self.refs[key] == 0:
            self.live -= self.size[key]
            self._record(self.owner[key])

    @staticmethod
    def _unpack(x):
        return x.t if isinstance(x, _Held) else x

    def run(self, forward) -> Timeline:
        """forward() returns the scalar loss. Runs forward and backward under the hooks."""
        handles = []
        for i, b in enumerate(self.blocks):
            handles.append(b.register_forward_pre_hook(lambda m, a, i=i: setattr(self, "tag", f"block {i}")))
            handles.append(b.register_forward_hook(lambda m, a, o: setattr(self, "tag", "head")))
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                loss = forward()
                self.tl.turn = len(self.tl.events) - 1
                self.phase = "backward"
                loss.backward()
                del loss
        finally:
            for h in handles:
                h.remove()
        return self.tl


def track(model: nn.Module, input_ids: torch.Tensor) -> Timeline:
    tracker = SavedTensorTracker(model, list(model.layers))
    return tracker.run(lambda: model(input_ids, labels=input_ids).loss)


def saved_elems_per_token_per_layer(tl: Timeline, tokens: int, hidden: int, bytes_per_el: int) -> float:
    """Average saved elements per token per block, in units of H, to compare with the post's 16H."""
    blocks = [v for k, v in tl.saved_by_tag.items() if k.startswith("block")]
    return sum(blocks) / len(blocks) / bytes_per_el / tokens / hidden


def ascii_mountain(tl: Timeline, width: int = 50, rows: int = 32) -> str:
    """One line per sampled event: live saved bytes as a bar. The turn is marked."""
    n = len(tl.events)
    idx = sorted(set([round(i * (n - 1) / (rows - 1)) for i in range(rows)] + [tl.turn]))
    peak = tl.peak or 1
    lines = []
    for i in idx:
        e = tl.events[i]
        bar = "#" * round(width * e.live_bytes / peak)
        mark = "  <- the turn: loss computed" if i == tl.turn else ""
        lines.append(f"{i:5d} {e.phase:8s} {e.tag:8s} {e.live_bytes / 2**20:8.2f} MiB |{bar}{mark}")
    return "\n".join(lines)
