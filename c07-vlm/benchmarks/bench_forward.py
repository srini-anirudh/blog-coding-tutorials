"""What does a picture cost? Forward latency, encoder share and KV-cache size against image tokens.

    python c07-vlm/benchmarks/bench_forward.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinyvlm import build_tinyvlm, describe_device, pick_device  # noqa: E402


def sync(device: torch.device) -> None:
    """GPU work is asynchronous; wait for it before reading the clock."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    transformers.logging.disable_progress_bar()
    device = pick_device(args.device)
    rows = []
    hardware = describe_device(device)
    print(f"device {hardware}, float32, one sample, 14 text tokens, best of {args.repeats}")
    print(f"{'crops':>5} {'s':>2} {'img tok':>7} {'seq':>5} {'encode ms':>10} {'lm ms':>8} {'total ms':>9} {'KV cache MB':>12} {'GPU peak MB':>12}")
    for factor in (2, 1):
        model, processor = build_tinyvlm(pixel_shuffle_factor=factor)
        model.to(device).eval()
        for crops in (1, 7):
            message = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]}
            prompt = processor.apply_chat_template([message], add_generation_prompt=True)
            batch = processor(prompt, [Image.new("RGB", (224, 224), "white")])
            # Build the crop count by hand so each row is exactly 1 or 7 crops.
            ids = batch["input_ids"][0].tolist()
            where = ids.index(processor.image_token_id)
            n = processor.image_seq_len
            ids = ids[:where] + [processor.image_token_id] * (n * crops) + ids[where + n :]
            input_ids = torch.tensor([ids], device=device)
            mask = torch.ones_like(input_ids)
            pixels = batch["pixel_values"].repeat(crops, 1, 1, 1).to(device)
            encode, total = [], []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    features = model.get_image_features(pixels)
                    sync(device)
                    encode.append(time.perf_counter() - start)
                    start = time.perf_counter()
                    out = model(input_ids, pixels, mask, use_cache=True)
                    sync(device)
                    total.append(time.perf_counter() - start)
            cache_mb = sum(k.numel() + v.numel() for k, v in ((layer.keys, layer.values) for layer in out.past_key_values.layers)) * 4 / 1e6
            enc, tot = min(encode) * 1000, min(total) * 1000
            peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
            row = {"crops": crops, "shuffle": factor, "image_tokens": n * crops, "seq": len(ids), "encode_ms": enc, "lm_ms": tot - enc, "total_ms": tot, "kv_cache_mb": cache_mb, "gpu_peak_mb": peak_mb}
            rows.append(row)
            peak = f"{peak_mb:.0f}" if peak_mb is not None else "-"
            print(f"{crops:>5} {factor:>2} {n * crops:>7} {len(ids):>5} {enc:>10.0f} {tot - enc:>8.0f} {tot:>9.0f} {cache_mb:>12.1f} {peak:>12}")
        del model
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"bench-forward-{device.type}.json").write_text(json.dumps({"hardware": hardware, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
