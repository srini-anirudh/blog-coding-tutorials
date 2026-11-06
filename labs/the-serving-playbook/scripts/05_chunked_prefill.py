"""Sections 8 and 9: one long prompt arrives while 32 sequences (by default) are decoding.

Unchunked, the whole prompt is one iteration and every decode waits for it. Chunked (Sarathi), the prompt
is split into pieces of at most C tokens per iteration, with the decodes riding along in every one.
Part B prefills the same long prompt alone at each chunk size: small chunks re-read the weights more often,
so prefill throughput falls.

Default model: llama3-8b-random, Llama-3-8B shapes with random weights (speed and memory do not depend on the
weight values; the text it produces is meaningless). Pass --model HuggingFaceTB/SmolLM2-360M for a real checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servekit.engine import Engine  # noqa: E402
from servekit.metrics import Report, device_and_dtype, percentile, random_prompts  # noqa: E402
from servekit.model import load_model  # noqa: E402
from servekit.scheduler import SchedulerConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-8b-random")
    ap.add_argument("--long", type=int, default=7680)
    ap.add_argument("--decodes", type=int, default=32)
    ap.add_argument("--chunks", default="256,512,1024,2048")
    args = ap.parse_args()
    dev, dtype = device_and_dtype()
    model = load_model(args.model, dtype, dev)
    v = model.config.vocab_size
    shorts = random_prompts(args.decodes, 128, 128, v, seed=1)
    long_prompt = random_prompts(1, args.long, args.long, v, seed=2)[0]
    chunks = [int(c) for c in args.chunks.split(",")]
    rep = Report("05_chunked_prefill.txt")
    rep(f"model {args.model}; {args.decodes} decoding sequences (128-token prompts, 300 output tokens);"
        f" a {args.long}-token prompt arrives 0.5 s in and asks for 16 tokens")
    rows = []
    for c in [None, *chunks]:
        cfg = SchedulerConfig(max_num_seqs=args.decodes + 1, max_num_batched_tokens=c or args.long + args.decodes,
                              chunked_prefill=c is not None)
        eng = Engine(model, num_blocks=4000, sched=cfg, max_model_len=8192)
        reqs = [(0.0, p, 300) for p in shorts] + [(0.5, long_prompt, 16)]
        eng.run([(0.0, p, 20) for p in shorts[:4]])  # warm-up
        eng.finished.clear()
        eng.stats.clear()
        eng.run(reqs)
        long_seq = max(eng.finished, key=lambda s: len(s.prompt))
        gaps = [b - a for s in eng.finished if s is not long_seq for a, b in zip(s.token_times, s.token_times[1:])]
        mixed = [st.seconds for st in eng.stats if st.num_prefill_tokens and st.num_decodes]
        rows.append({"chunk": "none" if c is None else c, "long_ttft_ms": (long_seq.token_times[0] - long_seq.arrival) * 1e3,
                     "decode_itl_p50_ms": percentile(gaps, 50) * 1e3, "decode_itl_p99_ms": percentile(gaps, 99) * 1e3,
                     "decode_itl_max_ms": max(gaps) * 1e3, "mixed_iters": len(mixed),
                     "mixed_iter_ms": (sum(mixed) / len(mixed) * 1e3) if mixed else 0.0})
    rep.table(rows, {k: "{:.1f}" for k in rows[0] if k.endswith("_ms")})
    rep(f"ITL = gap between consecutive tokens of the {args.decodes} decoding sequences, over the whole run.")
    rep("mixed_iter_ms = mean duration of iterations that held both prefill tokens and decodes.")

    rep("")
    rep(f"Part B: prefill the {args.long}-token prompt alone, chunk size C (one iteration per chunk)")
    prow = []
    for c in sorted({64, 128, *chunks, args.long}):
        eng = Engine(model, num_blocks=4000, sched=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=c),
                     max_model_len=8192)
        eng.run([(0.0, long_prompt[:512], 1)])  # warm-up
        eng.stats.clear()
        eng.run([(0.0, long_prompt, 1)])
        t = sum(st.seconds for st in eng.stats)
        prow.append({"C": c, "iterations": len(eng.stats), "prefill_ms": t * 1e3, "tok/s": args.long / t})
    rep.table(prow, {"prefill_ms": "{:.1f}", "tok/s": "{:.0f}"})
    rep.save({"interference": rows, "prefill_alone": prow})


if __name__ == "__main__":
    main()
