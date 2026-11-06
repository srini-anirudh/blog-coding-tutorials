"""Post sections 11 and 15: what batching buys, and why verifying gamma + 1 positions is nearly free.

Llama-3-8B shape, random bf16 weights, every step a CUDA-graph replay.

1. Batch sweep at a short context (512): the weight read is shared, so steps stay nearly flat while
   aggregate throughput rises.
2. Batch sweep at 8,192 context: every sequence brings a private cache, so the step grows with B, and the card
   runs out of memory near the post's B_max of 57.
3. Verification width: one step that scores R new positions per sequence (R = gamma + 1), at batch 1 and at
   larger batches, so B * R crosses the ridge.

    python labs/the-mechanics-of-llm-inference/scripts/05_batching.py --out .../results/batching_h100.txt
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferlab.decoder import SHAPES, Decoder  # noqa: E402
from inferlab.ledger import MODELS, decode_intensity  # noqa: E402
from inferlab.timing import Report, cuda_graph, measured_bandwidth, time_ms  # noqa: E402


def step_ms(model: Decoder, batch: int, rows: int, context: int) -> float | None:
    """Graph-replayed time of one step: `rows` new tokens per sequence after `context` cached ones."""
    model.k_cache = model.v_cache = None
    gc.collect()
    torch.cuda.empty_cache()
    try:
        model.allocate_cache(batch, context + rows)
        ids = torch.zeros(batch, rows, dtype=torch.long, device="cuda")
        replay = cuda_graph(lambda: model(ids, context))
        ms = time_ms(replay, warmup=3, iters=20)
        del replay
        return ms
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        model.k_cache = model.v_cache = None
        gc.collect()
        torch.cuda.empty_cache()


def kv_seq(model: Decoder, context: int) -> int:
    c = model.config
    return 2 * c.layers * c.n_kv * c.head_dim * 2 * context


def batch_sweep(rep: Report, model: Decoder, context: int, batches: list[int], bw: float) -> None:
    w = model.weight_bytes(read_per_decode_step=True)
    m = MODELS["llama-3-8b"]
    rep(f"\nBatch sweep at context {context}: KV per sequence {kv_seq(model, context) / 1e9:.3f} GB")
    rep(f"{'B':>4} {'step ms':>8} {'floor ms':>9} {'agg tok/s':>10} {'user tok/s':>10} {'KV share':>9} "
        f"{'I (post)':>9}")
    base = None
    for b in batches:
        ms = step_ms(model, b, 1, context)
        q = w + b * kv_seq(model, context)
        if ms is None:
            rep(f"{b:>4}  does not fit: needs {model.weight_bytes() / 1e9:.1f} GB weights + "
                f"{b * kv_seq(model, context) / 1e9:.1f} GB cache")
            continue
        base = base or ms
        rep(f"{b:>4} {ms:>8.2f} {q / bw * 1e3:>9.2f} {b * 1e3 / ms:>10.0f} {1e3 / ms:>10.0f} "
            f"{100 * (q - w) / q:>8.0f}% {decode_intensity(m, b, context):>9.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-context", type=int, default=512)
    ap.add_argument("--short-batches", default="1,4,16,64,128,256")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--batches", default="1,2,4,8,16,32,48,56,64")
    ap.add_argument("--verify-context", type=int, default=2048)
    ap.add_argument("--verify-rows", default="1,2,3,5,8,16,32,64")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    rep = Report(args.out)
    bw = measured_bandwidth()
    rep(f"Llama-3-8B shape, random bf16 weights, CUDA-graph replay. Measured copy bandwidth {bw / 1e12:.2f} TB/s.")
    rep("floor = (decode-read weights + B x KV) / measured bandwidth. I is the post's I(B), FLOP/byte.")
    model = Decoder.random(SHAPES["llama-3-8b"], max_len=2048, device="cuda", dtype=torch.bfloat16)

    batch_sweep(rep, model, args.short_context, [int(x) for x in args.short_batches.split(",")], bw)
    batch_sweep(rep, model, args.context, [int(x) for x in args.batches.split(",")], bw)

    n = args.verify_context
    rep(f"\nVerification width at context {n}: R new positions per sequence in one step (R = gamma + 1)")
    rep(f"{'B':>4} {'R':>4} {'B*R':>5} {'step ms':>8} {'vs R=1':>7}")
    for b, rows in [(1, r) for r in map(int, args.verify_rows.split(","))] + [(8, 1), (8, 5), (64, 1), (64, 5)]:
        ms = step_ms(model, b, rows, n)
        if rows == 1:
            one = ms
        rep(f"{b:>4} {rows:>4} {b * rows:>5} {ms:>8.2f} {ms / one:>6.2f}x")
    rep("post: verification costs about one step while B(gamma+1) stays below B* = 295 on an H100")
    rep.save()


if __name__ == "__main__":
    main()
