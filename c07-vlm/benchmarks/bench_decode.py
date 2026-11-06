"""What does the KV cache save? Greedy decoding with and without it, against prompt length.

    python c07-vlm/benchmarks/bench_decode.py

Without a cache, every new token re-runs the whole prompt, image tokens included. The longer the prompt
(more crops, or no pixel shuffle), the more that costs. Both timings include one encoder pass.
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
from tinyvlm import build_tinyvlm, describe_device  # noqa: E402
from tinyvlm.generation import greedy_generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-tokens", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    transformers.logging.disable_progress_bar()
    hardware = describe_device(torch.device("cpu"))
    print(f"device {hardware}, float32, {args.new_tokens} new tokens, best of {args.repeats}")
    print(f"{'crops':>5} {'s':>2} {'prompt':>7} {'cached s':>9} {'recomputed s':>13} {'ratio':>6}")
    message = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]}
    rows = []
    for factor in (2, 1):
        model, processor = build_tinyvlm(pixel_shuffle_factor=factor)
        batch = processor(processor.apply_chat_template([message], True), [Image.new("RGB", (224, 224), "white")])
        for crops in (1, 7):
            # Same trick as bench_forward.py: widen the single image to exactly `crops` crops.
            ids = batch["input_ids"][0].tolist()
            where, n = ids.index(processor.image_token_id), processor.image_seq_len
            ids = ids[:where] + [processor.image_token_id] * (n * crops) + ids[where + n :]
            input_ids = torch.tensor([ids])
            mask, pixels = torch.ones_like(input_ids), batch["pixel_values"].repeat(crops, 1, 1, 1)
            best, outputs = {}, {}
            for use_cache in (True, False):
                seconds = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    outputs[use_cache] = greedy_generate(model, input_ids, pixels, mask, args.new_tokens, use_cache=use_cache)
                    seconds.append(time.perf_counter() - start)
                best[use_cache] = min(seconds)
            same = torch.equal(outputs[True], outputs[False])
            rows.append({"crops": crops, "shuffle": factor, "prompt_tokens": len(ids), "cached_s": best[True],
                         "recomputed_s": best[False], "same_tokens": same})
            print(f"{crops:>5} {factor:>2} {len(ids):>7} {best[True]:>9.2f} {best[False]:>13.2f} "
                  f"{best[False] / best[True]:>6.1f}{'' if same else '  outputs differ'}", flush=True)
        del model
    (ROOT / "results" / "bench-decode-cpu.json").write_text(json.dumps({"hardware": hardware, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
