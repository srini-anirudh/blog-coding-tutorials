"""Section 15: a small model's decode step is launch-bound. Capture it once as a CUDA graph and replay it.

SmolLM2-360M, one decode step for B sequences at context 256. Eager runs the same paged forward pass kernel
by kernel from Python. The graph path pads B up to a captured bucket and pads every block table to
max_model_len, as a captured graph requires. Both paths must emit the same tokens.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from servekit.engine import Engine  # noqa: E402
from servekit.metrics import Report, random_prompts  # noqa: E402
from servekit.model import load_model  # noqa: E402
from servekit.scheduler import SchedulerConfig  # noqa: E402


def decode_times(model, b: int, ctx: int, graph: bool, steps: int = 40):
    eng = Engine(model, num_blocks=b * (ctx // 16 + 8), cuda_graph=graph, max_model_len=ctx + 64,
                 sched=SchedulerConfig(max_num_seqs=b, max_num_batched_tokens=b * ctx))
    seqs = [eng.add_request(p, steps + 1) for p in random_prompts(b, ctx, ctx, model.config.vocab_size, seed=b)]
    for _ in range(steps + 1):
        eng.step()
    dec = sorted(st.seconds for st in eng.stats[3:])  # skip prefill and the capture step
    return dec[len(dec) // 2], [s.output for s in seqs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--ctx", type=int, default=256)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU")
        return
    model = load_model(args.model, torch.bfloat16, "cuda")
    rep = Report("08_cuda_graphs.txt")
    rows = []
    for b in (1, 4, 16, 64, 128):
        te, oe = decode_times(model, b, args.ctx, False)
        tg, og = decode_times(model, b, args.ctx, True)
        same = sum(x == y for x, y in zip(oe, og)) / b
        rows.append({"B": b, "eager_ms": te * 1e3, "graph_ms": tg * 1e3, "speedup": te / tg,
                     "eager_tok/s": b / te, "graph_tok/s": b / tg, "same_tokens": same})
    rep(f"{args.model}, decode step at context {args.ctx}, median wall time per step (scheduling + forward + sampling)")
    rep.table(rows, {"eager_ms": "{:.2f}", "graph_ms": "{:.2f}", "speedup": "{:.2f}", "eager_tok/s": "{:.0f}",
                     "graph_tok/s": "{:.0f}", "same_tokens": "{:.2f}"})
    rep("same_tokens = fraction of sequences whose greedy outputs match between eager and graph (bf16).")
    rep.save({"rows": rows})


if __name__ == "__main__":
    main()
