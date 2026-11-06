"""The engine loop: schedule, build the ragged batch, run one forward pass, sample, repeat.

Sampling is greedy (argmax) and requests ignore end-of-sequence, as serving benchmarks usually do, so each
request produces exactly max_new_tokens tokens and runs are reproducible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .block_manager import BlockManager
from .model import BatchMeta, PagedKVCache, PagedLlama
from .scheduler import RUNNING, Scheduler, SchedulerConfig, Sequence


@dataclass
class StepStat:
    t_start: float
    seconds: float
    num_tokens: int  # token positions in this forward pass
    num_prefill_tokens: int
    num_decodes: int
    num_useful_decodes: int  # static mode: decodes of sequences that still want tokens
    kv_usage: float


class Engine:
    def __init__(
        self,
        model: PagedLlama,
        num_blocks: int,
        block_size: int = 16,
        sched: SchedulerConfig | None = None,
        enable_prefix_caching: bool = False,
        watermark: float = 0.01,
        cuda_graph: bool = False,
        max_model_len: int = 2048,
        shed_after: float | None = None,
        precapture: bool = False,
    ) -> None:
        p = next(model.parameters())
        self.model, self.device, self.dtype = model, p.device, p.dtype
        self.block_size = block_size
        self.cache = PagedKVCache(model.config, num_blocks, block_size, p.dtype, p.device)
        self.bm = BlockManager(num_blocks, block_size, enable_prefix_caching, watermark=watermark)
        self.bm.copy_fn = self.cache.copy_block
        self.sched = Scheduler(sched or SchedulerConfig(), self.bm)
        self.max_blocks = -(-max_model_len // block_size)
        self.graphs = CudaGraphDecoder(self) if cuda_graph else None
        if self.graphs is not None and precapture:
            self.graphs.capture_all()  # capture every bucket now, so no request pays for a capture
            # (otherwise each bucket is captured the first time a batch of that size decodes)
        self.stats: list[StepStat] = []
        self.finished: list[Sequence] = []
        self.shed: list[Sequence] = []
        # SLO-aware admission: drop a request that has queued longer than this, before it costs any compute.
        self.shed_after = shed_after
        self._next_id = 0

    def add_request(self, prompt: list[int], max_new_tokens: int, arrival: float | None = None, **kw) -> Sequence:
        seq = Sequence(self._next_id, list(prompt), max_new_tokens, time.perf_counter() if arrival is None else arrival, **kw)
        self._next_id += 1
        self.sched.add(seq)
        return seq

    # ---- one iteration ------------------------------------------------------------------------
    def _prefill_index(self, table: list[int], t: int, ctx: int):
        """Slots to gather (ctx,) and the attention mask for a segment of t new tokens at the end of ctx."""
        bs, dev = self.block_size, self.device
        tbl = torch.tensor(table[: -(-ctx // bs)], device=dev)
        slots = (tbl[:, None] * bs + torch.arange(bs, device=dev)[None]).flatten()[:ctx]
        if t == ctx:
            return slots, None
        qpos = torch.arange(ctx - t, ctx, device=dev)[:, None]
        return slots, torch.arange(ctx, device=dev)[None, :] <= qpos

    def build_meta(self, plan: list[tuple[Sequence, int]]) -> tuple[BatchMeta, list[Sequence]]:
        ids, pos, slots, logits_rows, samplers = [], [], [], [], []
        dec_rows, dec_tables, dec_ctx, prefills = [], [], [], []
        bs = self.block_size
        for seq, n in plan:
            start, end = seq.num_computed, seq.num_computed + n
            self.bm.prepare_write(seq.seq_id, start, end)
            table = self.bm.tables[seq.seq_id]
            row0 = len(ids)
            toks = seq.tokens
            ids.extend(toks[start:end])
            pos.extend(range(start, end))
            slots.extend(table[p // bs] * bs + p % bs for p in range(start, end))
            if n == 1 and start > 0:
                dec_rows.append(row0)
                dec_tables.append(table)
                dec_ctx.append(end)
            else:
                prefills.append((row0, row0 + n, end) + self._prefill_index(table, n, end))
            if end == seq.num_tokens:  # this sequence emits a token this iteration
                logits_rows.append(row0 + n - 1)
                samplers.append(seq)
        dev = self.device
        meta = BatchMeta(
            input_ids=torch.tensor(ids, device=dev),
            positions=torch.tensor(pos, device=dev),
            slot_mapping=torch.tensor(slots, device=dev),
            logits_rows=torch.tensor(logits_rows, device=dev, dtype=torch.long),
            prefills=prefills,
        )
        if dec_rows:
            width = max(len(t) for t in dec_tables)
            meta.dec_rows = torch.tensor(dec_rows, device=dev)
            meta.dec_tables = torch.tensor([t + [0] * (width - len(t)) for t in dec_tables], device=dev)
            meta.dec_ctx = torch.tensor(dec_ctx, device=dev)
        return meta, samplers

    def _shed(self, now: float) -> None:
        q = self.sched.waiting
        keep = [s for s in q if s.num_preemptions or now - s.arrival <= self.shed_after]
        if len(keep) < len(q):
            kept = {s.seq_id for s in keep}
            self.shed.extend(s for s in q if s.seq_id not in kept)
            q.clear()
            q.extend(keep)

    def step(self) -> list[Sequence]:
        t0 = time.perf_counter()
        if self.shed_after is not None:
            self._shed(t0)
        plan = self.sched.schedule()
        if not plan:
            return []
        all_decode = all(n == 1 and s.num_computed > 0 for s, n in plan)
        if self.graphs is not None and all_decode and len(plan) <= self.graphs.max_batch:
            samplers = [s for s, _ in plan]
            next_tokens = self.graphs.run(plan)
        else:
            meta, samplers = self.build_meta(plan)
            logits = self.model(meta, self.cache)
            next_tokens = logits.argmax(-1).tolist() if samplers else []  # .tolist() waits for the GPU
        now = time.perf_counter()
        done: list[Sequence] = []
        n_prefill = sum(n for s, n in plan if not (n == 1 and s.num_computed > 0))
        n_dec = len(plan) - sum(1 for s, n in plan if not (n == 1 and s.num_computed > 0))
        useful = sum(1 for s, n in plan if n == 1 and s.num_computed > 0 and not s.done_useful)
        for seq, n in plan:
            seq.num_computed += n
            self.bm.register_computed(seq.seq_id, seq.tokens, seq.num_computed, seq.extra_keys)
        for seq, tok in zip(samplers, next_tokens):
            seq.output.append(tok)
            if not seq.done_useful:
                seq.token_times.append(now)
                if len(seq.token_times) >= seq.max_new_tokens:
                    seq.done_useful = True
                    seq.finish_time = now
        if self.sched.cfg.mode == "static":
            # The whole group returns together, once its longest member is done.
            if self.sched.running and all(s.done_useful for s in self.sched.running):
                done = list(self.sched.running)
        else:
            done = [s for s in samplers if s.done_useful and s.state == RUNNING]
        for seq in done:
            self.sched.finish(seq)
        self.finished.extend(done)
        self.stats.append(StepStat(t0, now - t0, sum(n for _, n in plan), n_prefill, n_dec, useful, self.bm.usage()))
        return done

    def run(self, requests: list[tuple[float, list[int], int]] | None = None, max_steps: int = 10**7) -> float:
        """Serve requests (arrival offset in seconds, prompt, max_new_tokens) in real time. Returns wall seconds."""
        pending = sorted(requests or [], key=lambda r: r[0])
        t0 = time.perf_counter()
        k = 0
        for _ in range(max_steps):
            now = time.perf_counter() - t0
            while k < len(pending) and pending[k][0] <= now:
                off, prompt, n = pending[k]
                self.add_request(prompt, n, arrival=t0 + off)
                k += 1
            if not self.sched.has_work():
                if k == len(pending):
                    break
                time.sleep(max(0.0, min(0.002, pending[k][0] - now)))
                continue
            self.step()
        return time.perf_counter() - t0


class CudaGraphDecoder:
    """Capture one CUDA graph per decode batch size bucket and replay it.

    Graphs pin shapes and addresses, so every input lives in a preallocated buffer, block tables are padded
    to max_blocks, and the batch is padded up to the next bucket. Padded rows write into the scratch block.
    """

    BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self, engine: Engine, max_batch: int = 128) -> None:
        self.e = engine
        self.max_batch = max_batch
        dev, B, nb = engine.device, max_batch, engine.max_blocks
        self.ids = torch.zeros(B, dtype=torch.long, device=dev)
        self.pos = torch.zeros(B, dtype=torch.long, device=dev)
        self.slots = torch.full((B,), engine.cache.scratch_slot, dtype=torch.long, device=dev)
        self.tables = torch.zeros(B, nb, dtype=torch.long, device=dev)
        self.ctx = torch.ones(B, dtype=torch.long, device=dev)
        self.rows = torch.arange(B, device=dev)
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.pool = None

    def _meta(self, b: int) -> BatchMeta:
        r = self.rows[:b]
        return BatchMeta(self.ids[:b], self.pos[:b], self.slots[:b], r, r, self.tables[:b], self.ctx[:b])

    def capture_all(self) -> None:
        for b in self.BUCKETS:
            if b <= self.max_batch:
                self.e.model(self._meta(b), self.e.cache)  # one eager pass on the default stream first, so the
                torch.cuda.synchronize()  # cuBLAS handles and workspaces exist before capture begins
                self.capture(b)

    def capture(self, b: int) -> None:
        meta = self._meta(b)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):  # warm up kernels and the allocator outside the graph
                self.e.model(meta, self.e.cache)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self.e.model(meta, self.e.cache).argmax(-1)
        self.pool = g.pool()
        self.graphs[b] = (g, out)

    def run(self, plan: list[tuple[Sequence, int]]) -> list[int]:
        n = len(plan)
        b = next(x for x in self.BUCKETS if x >= n)
        bs = self.e.block_size
        ids, pos, slots, ctx = [], [], [], []
        tables = torch.zeros(b, self.e.max_blocks, dtype=torch.long)
        for i, (seq, _) in enumerate(plan):
            p = seq.num_computed
            self.e.bm.prepare_write(seq.seq_id, p, p + 1)
            t = self.e.bm.tables[seq.seq_id]
            ids.append(seq.tokens[p])
            pos.append(p)
            slots.append(t[p // bs] * bs + p % bs)
            ctx.append(p + 1)
            tables[i, : len(t)] = torch.tensor(t)
        pad = b - n
        self.ids[:b].copy_(torch.tensor(ids + [0] * pad))
        self.pos[:b].copy_(torch.tensor(pos + [0] * pad))
        self.slots[:b].copy_(torch.tensor(slots + [self.e.cache.scratch_slot] * pad))
        self.ctx[:b].copy_(torch.tensor(ctx + [1] * pad))
        self.tables[:b].copy_(tables)
        if b not in self.graphs:
            self.capture(b)
        g, out = self.graphs[b]
        g.replay()
        return out[:n].tolist()
