"""Sections 5, 6 and 21: KV bytes per token, the allocation problem, paging, and sharing by reference count.

CPU only. The pool is sized like the post's example: about 60 GB of KV for Llama-3-8B at 128 KiB per token.
The request trace is synthetic (prompts 50 to 1,000 tokens, outputs log-uniform from 16 to 2,000, clients
ask for max_tokens 2,048). These are simulation results, not measurements of any real system.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servekit.alloc_sim import Req, sharing_blocks, simulate  # noqa: E402
from servekit.metrics import Report  # noqa: E402
from servekit.model import get_config, kv_bytes_per_token  # noqa: E402


def main() -> None:
    rep = Report("02_kv_memory.txt")
    for name in ("llama3-8b", "smollm2-135m"):
        c = get_config(name)
        b = kv_bytes_per_token(c, 2)
        rep(f"{name}: KV per token = 2 x {c.num_layers} x {c.num_kv_heads} x {c.head_dim} x 2 B = {b / 1024:.0f} KiB;"
            f" an 8,192-token sequence holds {b * 8192 / 2**30:.2f} GiB")

    kv_per_tok = kv_bytes_per_token(get_config("llama3-8b"), 2)
    capacity = int(60e9 // kv_per_tok)
    rng = random.Random(0)
    reqs = [Req(rng.randint(50, 1000), int(16 * (2000 / 16) ** rng.random()), 2048) for _ in range(3000)]
    mean_out = sum(r.output for r in reqs) / len(reqs)
    rep("")
    rep(f"Pool: 60 GB / 128 KiB = {capacity:,} tokens. Trace: {len(reqs)} requests, mean output {mean_out:.0f} tokens,"
        " max_model_len 8,192, block size 16.")
    rows = [simulate(reqs, capacity, p, max_model_len=8192) for p in ("reserve_max", "contiguous", "paged")]
    base = rows[0]["steps"]
    for r in rows:
        r["speedup_vs_reserve_max"] = base / r["steps"]
    rep.table(rows, {"mean_concurrency": "{:.1f}", "kv_utilization": "{:.3f}", "speedup_vs_reserve_max": "{:.2f}"})
    rep("kv_utilization = tokens actually written / tokens reserved, averaged over steps.")
    rep("frag_blocked_steps = steps where the next request had enough total free memory but no hole big enough.")
    rep("steps = decode steps to drain the trace, a proxy for time at a fixed step cost.")

    rep("")
    rep("Parallel sampling, n = 8 samples of a 2,000-token prompt, after each sample writes 1 token:")
    s = sharing_blocks(2000, 8, 1)
    mib = 16 * kv_per_tok / 2**20
    rep(f"  shared blocks {s['shared_blocks']} ({s['shared_blocks'] * mib:.0f} MiB) versus private copies "
        f"{s['private_blocks']} ({s['private_blocks'] * mib:.0f} MiB); the prompt is {s['prompt_blocks']} blocks")
    s2 = sharing_blocks(2000, 8, 100)
    rep(f"  after 100 tokens each: shared {s2['shared_blocks']} blocks versus private {s2['private_blocks']}")
    rep.save({"simulation": rows, "sharing_1tok": s, "sharing_100tok": s2, "capacity_tokens": capacity})


if __name__ == "__main__":
    main()
