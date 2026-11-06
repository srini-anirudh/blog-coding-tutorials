"""The attention zoo: training, prefill and decode for each architecture.

Run:  python c03-attention/benchmarks/bench_zoo.py [--device cpu] [--lengths 512 1024 ...] [--out file.json]
One attention layer, D = 256, Hq = 8, dh = 32. Each cell runs in a fresh process.
The GQA flash row exists only on CUDA, which runs in bf16; CPU runs in fp32.
On CUDA, cuDNN attention is switched off unless you pass --cudnn. PyTorch picks it by default on an H100,
and it spends 60 to 80 ms setting up each new shape. Prefill and every decode step are new shapes, so
with it on, the sdpa rows time that setup and not the attention.
At N = 16384 the naive, MLA and linear rows need tens of GiB: pass shorter --lengths on a small machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common
import torch

from attnzoo.config import AttentionConfig
from attnzoo.device import synchronize
from attnzoo.library import build_attention, make_cache

D, H = 256, 8
ZOO: dict[str, dict] = {
    "MHA": dict(attention_type="mha"),
    "GQA Hkv=2": dict(attention_type="gqa", num_kv_heads=2),
    "MQA": dict(attention_type="mqa"),
    "MLA r=64": dict(attention_type="mla", kv_lora_rank=64, qk_rope_head_dim=8),
    "local mask": dict(attention_type="gqa", num_kv_heads=2, window_size=256),
    "local chunked": dict(attention_type="gqa", num_kv_heads=2, window_size=256, backend="window_chunked"),
    "linear": dict(attention_type="linear"),
    "GQA naive": dict(attention_type="gqa", num_kv_heads=2, backend="naive"),
}
CUDA_ONLY: dict[str, dict] = {"GQA flash": dict(attention_type="gqa", num_kv_heads=2, backend="flash")}
DECODE_STEPS = 16


def child(a: dict) -> None:
    device = torch.device(a["device"])
    dtype = _common.dtype_for(a["device"])
    if device.type == "cuda" and not a["cudnn"]:
        torch.backends.cuda.enable_cudnn_sdp(False)  # SDPA then picks FlashAttention-2 or memory-efficient
    n = a["n"]
    cfg = AttentionConfig(hidden_size=D, num_heads=H, max_seq_len=n + DECODE_STEPS + 1, **{**ZOO, **CUDA_ONLY}[a["name"]])
    model = build_attention(cfg).to(device=device, dtype=dtype)
    x = torch.randn(1, n, D, device=device, dtype=dtype)
    target = torch.randn(1, n, D, device=device, dtype=dtype)
    repeats = 3 if n <= 4096 else 1

    # Training: forward and backward, separately timed. The first step is a warm-up and is not counted.
    fwd, bwd = [], []
    with _common.PeakMemory(device) as mem:
        for step_index in range(repeats + 1):
            synchronize(device)
            t0 = _common.time.perf_counter()
            loss = (model(x) - target).pow(2).mean()
            synchronize(device)
            t1 = _common.time.perf_counter()
            loss.backward()
            synchronize(device)
            t2 = _common.time.perf_counter()
            if step_index > 0:
                fwd.append((t1 - t0) * 1e3)
                bwd.append((t2 - t1) * 1e3)
            model.zero_grad(set_to_none=True)
    del loss

    # Prefill then decode. The linear layer's parallel form has no cache, so prefill it step by step
    # only for short lengths; its recurrence is what decoding uses.
    model.eval()
    with torch.inference_mode():
        cache = make_cache(cfg)
        synchronize(device)
        t0 = _common.time.perf_counter()
        model(x, kv_cache=cache)
        synchronize(device)
        prefill_ms = (_common.time.perf_counter() - t0) * 1e3
        step = torch.randn(1, 1, D, device=device, dtype=dtype)
        synchronize(device)
        t0 = _common.time.perf_counter()
        for _ in range(DECODE_STEPS):
            model(step, kv_cache=cache)
        synchronize(device)
        decode_ms = (_common.time.perf_counter() - t0) * 1e3 / DECODE_STEPS
    print(json.dumps({
        "fwd": _common.statistics.median(fwd), "bwd": _common.statistics.median(bwd), "peak": mem.bytes,
        "prefill_tok_s": n / (prefill_ms / 1e3), "prefill_ms": prefill_ms, "decode": decode_ms, "cache": cache.nbytes(),
        "params": sum(p.numel() for p in model.parameters()),
    }))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--child")
    p.add_argument("--device", nargs="+", default=["cpu"], help="cpu, mps or cuda")
    p.add_argument("--lengths", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--cudnn", action="store_true", help="CUDA: leave cuDNN attention on, as PyTorch's default does")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "bench_zoo.json")
    args = p.parse_args()
    if args.child:
        return child(json.loads(args.child))

    rows = []
    for device in args.device:
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda needs a CUDA GPU, and none is visible. Nothing was measured.")
        note = _common.hardware_note(device)
        cudnn = f", cuDNN attention {'on' if args.cudnn else 'off'}" if device == "cuda" else ""
        print(f"\n{note}. One layer, D = {D}, Hq = {H}, dh = 32, {_common.dtype_for(device)}{cudnn}.")
        print(f"{'layer':15}{'N':>6}{'fwd ms':>10}{'bwd ms':>10}{'peak MB':>9}{'prefill ms':>12}{'tok/s':>12}{'decode ms/tok':>15}{'cache MB':>10}")
        for name in [*ZOO, *(CUDA_ONLY if device == "cuda" else [])]:
            for n in args.lengths:
                r = _common.run_child(__file__, {"device": device, "n": n, "name": name, "cudnn": args.cudnn}, timeout=600)
                if "error" in r:
                    print(f"{name:15}{n:>6}  {r['error']}", flush=True)
                    continue
                print(f"{name:15}{n:>6}{r['fwd']:>10.1f}{r['bwd']:>10.1f}{r['peak'] / 2**20:>9.0f}{r['prefill_ms']:>12.1f}"
                      f"{r['prefill_tok_s']:>12,.0f}{r['decode']:>15.2f}{r['cache'] / 2**20:>10.2f}", flush=True)
                rows.append({"hardware": note, "device": device, "cudnn": args.cudnn, "name": name, "n": n, **r})
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
