"""Three ways to hold KV memory, simulated on one request trace (no model, only token counts).

reserve_max   one contiguous region of max_model_len tokens per sequence, reserved at admission.
contiguous    one contiguous region of (prompt + requested max_tokens) per sequence, placed first-fit in a
              linear address space. Sizes differ, so the free space breaks into holes.
paged         fixed-size blocks, taken one at a time as tokens are written (servekit.BlockManager).

Every step, each running sequence writes one token. A sequence that finishes frees its memory, and the
FIFO queue admits as many waiting requests as fit. Time is counted in decode steps.
"""

from __future__ import annotations

from dataclasses import dataclass

from .block_manager import BlockManager


@dataclass
class Req:
    prompt: int
    output: int  # tokens it will actually generate
    requested: int  # max_tokens the client asked for (an upper bound the server must respect)


class FirstFit:
    """A linear address space of `size` tokens with first-fit placement."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.used: dict[int, tuple[int, int]] = {}  # id -> (start, length)

    def alloc(self, rid: int, length: int) -> bool:
        start = 0
        for s, n in sorted(self.used.values()):
            if s - start >= length:
                break
            start = max(start, s + n)
        if start + length > self.size:
            return False
        self.used[rid] = (start, length)
        return True

    def free(self, rid: int) -> None:
        del self.used[rid]

    def largest_hole(self) -> int:
        best, start = 0, 0
        for s, n in sorted(self.used.values()):
            best, start = max(best, s - start), s + n
        return max(best, self.size - start)


def simulate(reqs: list[Req], capacity_tokens: int, policy: str, max_model_len: int, block_size: int = 16) -> dict:
    waiting = list(range(len(reqs)))
    running: dict[int, int] = {}  # id -> tokens written so far
    ff = FirstFit(capacity_tokens) if policy in ("reserve_max", "contiguous") else None
    bm = BlockManager(capacity_tokens // block_size, block_size, watermark=0.01) if policy == "paged" else None
    steps = conc_sum = 0
    used_sum = reserved_sum = 0.0
    preemptions = 0
    frag_blocked = 0  # steps where the head of the queue had enough total free memory but no hole
    while waiting or running:
        # Admit in FIFO order.
        while waiting:
            rid = waiting[0]
            r = reqs[rid]
            if policy == "paged":
                bm.allocate_prompt(rid, [0] * r.prompt)
                ok = bm.ensure(rid, r.prompt + 1, reserve=bm.watermark_blocks)
                if not ok:
                    bm.free_seq(rid)
            else:
                need = max_model_len if policy == "reserve_max" else r.prompt + r.requested
                ok = ff.alloc(rid, need)
                if not ok and capacity_tokens - sum(n for _, n in ff.used.values()) >= need:
                    frag_blocked += 1
            if not ok:
                break
            waiting.pop(0)
            running[rid] = r.prompt + 1
        # One decode step.
        steps += 1
        conc_sum += len(running)
        used_sum += sum(running.values())
        if policy == "paged":
            reserved_sum += (bm.num_blocks - bm.num_free) * block_size
        else:
            reserved_sum += sum(n for _, n in ff.used.values())
        for rid in list(running):
            if rid not in running:  # preempted earlier in this step
                continue
            r = reqs[rid]
            if running[rid] >= r.prompt + r.output:
                del running[rid]
                bm.free_seq(rid) if bm else ff.free(rid)
                continue
            running[rid] += 1
            while bm and rid in running and not bm.ensure(rid, running[rid]):
                # Out of blocks: preempt the newest sequence by recompute (its tokens are written again later).
                victim = max(running)
                del running[victim]
                bm.free_seq(victim)
                waiting.insert(0, victim)
                waiting.sort()
                preemptions += 1
    return {
        "policy": policy,
        "steps": steps,
        "mean_concurrency": conc_sum / steps,
        "kv_utilization": used_sum / max(1.0, reserved_sum),  # written tokens / reserved tokens
        "frag_blocked_steps": frag_blocked,
        "preemptions": preemptions,
    }


def sharing_blocks(prompt: int, n: int, generated: int, block_size: int = 16) -> dict:
    """Parallel sampling with n samples: blocks held with copy-on-write sharing versus n private copies."""
    bm = BlockManager(10**6, block_size)
    bm.allocate_prompt(0, [0] * prompt)
    bm.ensure(0, prompt)
    for i in range(1, n):
        bm.fork(0, i)
    for i in range(n):  # every sample writes `generated` new tokens
        for p in range(prompt, prompt + generated):
            bm.ensure(i, p + 1)
            bm.prepare_write(i, p, p + 1)
    shared = bm.num_blocks - bm.num_free
    private = n * bm.blocks_for(prompt + generated)
    return {"shared_blocks": shared, "private_blocks": private, "prompt_blocks": bm.blocks_for(prompt)}
