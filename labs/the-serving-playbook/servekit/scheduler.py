"""The iteration-level scheduler: which sequences run in the next forward pass, and with how many tokens.

It allocates the four scarce things the post lists: a token budget per iteration, sequence slots,
KV blocks (through the BlockManager), and priority (here: arrival order, FIFO).

mode="continuous"  Orca-style. Every iteration is rebuilt from whatever is runnable. A finished
                   sequence leaves at once and a waiting one takes its place on the next step.
mode="static"      The baseline. Admit a group of up to max_num_seqs, run it until its longest member
                   finishes, and only then admit the next group. Finished members keep their slot and keep
                   computing (slot squatting): the batch is a rectangle.

chunked_prefill=True splits a prompt across iterations so that no iteration exceeds the token budget
(Sarathi). With False, a prompt is prefilled in one iteration, however long, alongside the decodes.

When a running sequence needs a block and none is free, the newest running sequence is preempted by
recompute: its blocks are freed and it goes back to the front of the queue, to be prefilled again later
(prompt plus the tokens it had generated so far).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockManager

WAITING, RUNNING, FINISHED = "waiting", "running", "finished"


@dataclass
class Sequence:
    seq_id: int
    prompt: list[int]
    max_new_tokens: int
    arrival: float = 0.0
    extra_keys: list | None = None  # per-block extra hash keys (image hashes for a VLM prompt)
    output: list[int] = field(default_factory=list)
    num_computed: int = 0  # positions whose K and V are in the cache
    state: str = WAITING
    token_times: list[float] = field(default_factory=list)
    num_preemptions: int = 0
    done_useful: bool = False  # static mode: reached max_new_tokens but still squatting in the batch
    finish_time: float | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.output

    @property
    def num_tokens(self) -> int:
        return len(self.prompt) + len(self.output)


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    chunked_prefill: bool = True
    mode: str = "continuous"  # or "static"


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, bm: BlockManager) -> None:
        self.cfg = cfg
        self.bm = bm
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0
        self.prompt_tokens = 0  # tokens of every admitted prompt (re-admissions after preemption count again)
        self.cached_tokens = 0  # of those, found in the prefix cache

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, seq: Sequence) -> None:
        self.bm.free_seq(seq.seq_id)
        seq.num_computed = 0
        seq.state = WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    def finish(self, seq: Sequence) -> None:
        self.bm.free_seq(seq.seq_id)
        seq.state = FINISHED
        self.running.remove(seq)

    def schedule(self) -> list[tuple[Sequence, int]]:
        """Return [(sequence, tokens to compute this iteration)]."""
        cfg, bm = self.cfg, self.bm
        budget = cfg.max_num_batched_tokens
        out: list[tuple[Sequence, int]] = []
        preempted = False

        # 1. Running sequences first: decodes, and prompts whose chunked prefill is unfinished.
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_tokens - seq.num_computed, budget)
            while not bm.ensure(seq.seq_id, seq.num_computed + n):
                victim = self.running.pop()  # lowest priority: the newest arrival
                self._preempt(victim)
                preempted = True
                if victim is seq:
                    break
            if seq.state != RUNNING:
                break
            out.append((seq, n))
            budget -= n
            i += 1

        # 2. Admit waiting sequences, if nothing was just preempted (otherwise we would thrash).
        if cfg.mode == "static" and self.running:
            return out
        while self.waiting and budget > 0 and len(self.running) < cfg.max_num_seqs and not preempted:
            seq = self.waiting[0]
            if seq.seq_id not in bm.tables:
                seq.num_computed = bm.allocate_prompt(seq.seq_id, seq.tokens, seq.extra_keys)
            remaining = seq.num_tokens - seq.num_computed
            n = min(remaining, budget) if cfg.chunked_prefill else remaining
            if not cfg.chunked_prefill and n > budget and out:
                break  # an unchunked prompt waits for an iteration with room for all of it
            # Watermark: do not admit a request that would leave running sequences no room to grow.
            if not bm.ensure(seq.seq_id, seq.num_computed + n, reserve=bm.watermark_blocks):
                bm.free_seq(seq.seq_id)  # undo; cached blocks go back to the free list, hashes intact
                seq.num_computed = 0
                break
            self.waiting.popleft()
            self.prompt_tokens += seq.num_tokens
            self.cached_tokens += seq.num_computed
            seq.state = RUNNING
            self.running.append(seq)
            out.append((seq, n))
            budget -= n
        return out
