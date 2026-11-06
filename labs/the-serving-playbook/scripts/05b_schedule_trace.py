"""Section 8 as a schedule, with no model and no timing: tokens per iteration while a long prompt arrives.

32 sequences are decoding when a 7,680-token prompt joins. The scheduler is run on its own (the block manager
and the queue; nothing is computed) and we print what each iteration would contain. This is the part of
chunked prefill that is a scheduling fact, independent of how fast any GPU runs the iterations.
Time per iteration is what 05_chunked_prefill.py measures.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servekit.block_manager import BlockManager  # noqa: E402
from servekit.metrics import Report  # noqa: E402
from servekit.scheduler import Scheduler, SchedulerConfig, Sequence  # noqa: E402


def trace(chunk: int | None, long_len: int = 7680, decodes: int = 32, out_len: int = 300) -> list[tuple[int, int, int]]:
    """Return (tokens in iteration, tokens of the long prompt, decode sequences) for every iteration."""
    bm = BlockManager(8000, 16)
    cfg = SchedulerConfig(max_num_seqs=decodes + 1, max_num_batched_tokens=chunk or long_len + decodes,
                          chunked_prefill=chunk is not None)
    sch = Scheduler(cfg, bm)
    for i in range(decodes):
        sch.add(Sequence(i, [1] * 128, out_len))
    rows, joined = [], False
    while sch.has_work():
        if not joined and len(rows) == 3:  # the long prompt arrives after three iterations
            sch.add(Sequence(decodes, [2] * long_len, 16))
            joined = True
        plan = sch.schedule()
        n_dec = sum(1 for s, n in plan if n == 1 and s.num_computed > 0)
        rows.append((sum(n for _, n in plan), sum(n for s, n in plan if s.seq_id == decodes), n_dec))
        for seq, n in plan:
            seq.num_computed += n
            if seq.num_computed == seq.num_tokens:
                seq.output.append(0)
                if len(seq.output) >= seq.max_new_tokens:
                    sch.finish(seq)
    return rows


def main() -> None:
    rep = Report("05b_schedule_trace.txt")
    rep("A 7,680-token prompt joins 32 decoding sequences after iteration 3. Iterations that carry it:")
    summary = {}
    for chunk in (None, 2048, 512, 256):
        rows = trace(chunk)
        carrying = [r for r in rows if r[1] > 1]  # iterations that prefill part of the long prompt
        name = "no chunking" if chunk is None else f"chunk {chunk}"
        rep(f"  {name:12s}: largest iteration {max(r[0] for r in rows):5d} tokens; {len(carrying):3d} iterations carry the long"
            f" prompt, holding {min(r[2] for r in carrying)} to {max(r[2] for r in carrying)} decodes each")
        summary[name] = {"largest_iteration_tokens": max(r[0] for r in rows), "iterations_with_long_prompt": len(carrying)}
    rep("Chunking caps the iteration size, which caps how long the decodes wait for their next token.")
    rep("How long an iteration of C tokens takes is measured in 05_chunked_prefill.py.")
    rep.save(summary)


if __name__ == "__main__":
    main()
