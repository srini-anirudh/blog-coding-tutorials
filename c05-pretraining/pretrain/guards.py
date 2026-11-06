"""Failure detection and recovery for the training loop.

    bad batch          check_batch       token ids outside [0, vocab) or wrong shape
    NaN / Inf          train_step skips the update; NonFiniteGuard escalates to a rollback
    loss spike         SpikeDetector     loss far above its recent running statistics
    OOM                step_with_oom_retry   split micro-batches, same global batch
    corrupt checkpoint latest_valid_checkpoint (checkpoint.py) falls back to an older one
    dead process       torchrun --max-restarts, then auto-resume (trainer.py)
"""

from __future__ import annotations

import math

import torch


class BadBatchError(ValueError):
    pass


class TrainingDiverged(RuntimeError):
    pass


def check_batch(x: torch.Tensor, y: torch.Tensor, vocab_size: int, ignore: int = -100) -> None:
    """Fail loudly on a batch the model must never see.

    An id >= vocab_size raises an index error on CPU and CUDA, but on some backends an
    out-of-range embedding lookup reads garbage memory instead of failing.
    """
    if x.shape != y.shape or x.dim() != 2:
        raise BadBatchError(f"shape mismatch: x {tuple(x.shape)} y {tuple(y.shape)}")
    if x.dtype != torch.long:
        raise BadBatchError(f"input dtype {x.dtype}, expected int64")
    lo, hi = int(x.min()), int(x.max())
    if lo < 0 or hi >= vocab_size:
        raise BadBatchError(f"input ids in [{lo}, {hi}], vocab is [0, {vocab_size})")
    valid = y[y != ignore]
    if valid.numel() and (int(valid.min()) < 0 or int(valid.max()) >= vocab_size):
        raise BadBatchError(f"target ids outside [0, {vocab_size})")


class SpikeDetector:
    """Flag a loss more than `k` standard deviations above its exponential moving average."""

    def __init__(self, k: float = 6.0, beta: float = 0.98, warmup: int = 50):
        self.k, self.beta, self.warmup = k, beta, warmup
        self.mean, self.var, self.n = 0.0, 0.0, 0

    def update(self, loss: float) -> bool:
        if not math.isfinite(loss):
            return True
        if self.n == 0:
            self.mean = loss  # start the average at the first value, not at zero
        spike = self.n >= self.warmup and loss > self.mean + self.k * math.sqrt(self.var) + 1e-3
        if not spike:  # spikes do not pollute the statistics
            delta = loss - self.mean
            self.mean += (1 - self.beta) * delta
            self.var = self.beta * (self.var + (1 - self.beta) * delta * delta)
            self.n += 1
        return spike

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "n": self.n}

    def load_state_dict(self, s: dict) -> None:
        self.mean, self.var, self.n = s["mean"], s["var"], s["n"]


class NonFiniteGuard:
    """One skipped step is noise. `patience` in a row means the run has diverged."""

    def __init__(self, patience: int = 3):
        self.patience, self.consecutive, self.total = patience, 0, 0

    def update(self, skipped: bool) -> None:
        if not skipped:
            self.consecutive = 0
            return
        self.consecutive += 1
        self.total += 1
        if self.consecutive >= self.patience:
            raise TrainingDiverged(f"{self.consecutive} non-finite steps in a row")


def is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()


def split_micro_batches(micro: list[tuple[torch.Tensor, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Halve every micro-batch. Same sequences, twice the accumulation steps."""
    out = []
    for x, y in micro:
        if x.size(0) < 2:
            raise RuntimeError("cannot split a micro-batch of one sequence any further")
        h = x.size(0) // 2
        out += [(x[:h], y[:h]), (x[h:], y[h:])]
    return out


def step_with_oom_retry(step_fn, micro, opt, empty_cache, max_splits: int = 3):
    """Run step_fn(micro); on OOM free memory, split the micro-batches and retry.

    Because train_step normalises by the step's total token count, the split step has
    the same gradient as the original one: only memory use and speed change.
    """
    for attempt in range(max_splits + 1):
        try:
            return step_fn(micro), len(micro)
        except (torch.OutOfMemoryError, RuntimeError) as err:
            if not is_oom(err) or attempt == max_splits:
                raise
            opt.zero_grad(set_to_none=True)  # drop partial gradients from the failed attempt
            empty_cache()
            micro = split_micro_batches(micro)
            print(f"[oom] retrying with {len(micro)} micro-batches of {micro[0][0].size(0)}")
