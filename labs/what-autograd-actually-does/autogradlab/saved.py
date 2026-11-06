"""See what autograd keeps: every tensor a backward rule saves, how big it is, and how long it lives.

`SavedTensorLedger` uses `torch.autograd.graph.saved_tensors_hooks`. The pack hook runs each time a backward
node saves a tensor. It wraps the tensor in a small holder object and adds its bytes to a running total. A
finalizer on the holder subtracts them again when autograd frees the node. So `live` is the bytes held for the
backward pass right now, and `peak` is the most ever held at once.

Storage shared between two saves (the output of one op saved as the input of the next) is counted once.
"""

from __future__ import annotations

import weakref

import torch


class _Held:
    __slots__ = ("tensor", "__weakref__")

    def __init__(self, tensor):
        self.tensor = tensor


class SavedTensorLedger:
    def __init__(self, exclude=()):
        self.exclude = {t.untyped_storage().data_ptr() for t in exclude}
        self.live = 0
        self.peak = 0
        self.saves = 0
        self._refs: dict[int, int] = {}  # storage pointer -> how many holders reference it
        self.extra = 0  # bytes held outside autograd (checkpoint boundaries); added by callers

    def _pack(self, t: torch.Tensor):
        self.saves += 1
        key = t.untyped_storage().data_ptr()
        if key in self.exclude:
            return _Held(t)
        if self._refs.get(key, 0) == 0:
            self.live += t.untyped_storage().nbytes()
            self.peak = max(self.peak, self.live + self.extra)
        self._refs[key] = self._refs.get(key, 0) + 1
        held = _Held(t)
        weakref.finalize(held, self._release, key, t.untyped_storage().nbytes())
        return held

    def _release(self, key: int, nbytes: int) -> None:
        self._refs[key] -= 1
        if self._refs[key] == 0:
            del self._refs[key]
            self.live -= nbytes

    @staticmethod
    def _unpack(held: _Held):
        return held.tensor

    def add_extra(self, nbytes: int) -> None:
        self.extra += nbytes
        self.peak = max(self.peak, self.live + self.extra)

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)


def saved_bytes(fn, *inputs, exclude=()) -> tuple[int, int]:
    """Run `fn(*inputs)` and return (bytes saved for backward, number of saves).

    Tensors in `exclude` (usually the parameters, which exist anyway) are not counted.
    """
    with SavedTensorLedger(exclude) as ledger:
        out = fn(*inputs)
    nbytes = ledger.live
    del out
    return nbytes, ledger.saves


def graph_nodes(t: torch.Tensor) -> int:
    """How many backward nodes autograd recorded behind `t` (AccumulateGrad leaves included)."""
    if t.grad_fn is None:
        return 0
    seen, stack = set(), [t.grad_fn]
    while stack:
        f = stack.pop()
        if f is None or id(f) in seen:
            continue
        seen.add(id(f))
        stack += [g for g, _ in f.next_functions]
    return len(seen)
