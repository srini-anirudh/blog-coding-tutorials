"""Full fine-tuning vs LoRA: peak memory and seconds per optimizer step, each mode in its own process.

Peak resident memory only ever goes up, so one generation-heavy eval would hide the training number. That is why
each mode gets a fresh process that does nothing except load the model and take optimizer steps.

    python benchmarks/bench_memory.py --steps 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.chat import load_tokenizer  # noqa: E402
from posttrain.lora import apply_lora  # noqa: E402
from posttrain.models import load_policy  # noqa: E402
from posttrain.sft import batches, build_examples, masked_lm_loss  # noqa: E402
from posttrain.tasks import split_problems  # noqa: E402
from posttrain.utils import (  # noqa: E402
    BASE_MODEL,
    count_parameters,
    device_memory_mb,
    hardware_label,
    pick_device,
    polite_threads,
    save_json,
    synchronize,
)


def worker(mode: str, steps: int, batch_size: int) -> dict:
    """Load, then train for `steps`. Returns memory after loading and at the peak."""
    polite_threads()
    device = pick_device()
    tok = load_tokenizer()
    train, _ = split_problems(500, 10, seed=0, follow_up_frac=0.3)
    stream = batches(build_examples(tok, train), batch_size, tok.pad_token_id, False, 256, 0)
    model = load_policy(BASE_MODEL, device).train()
    if mode == "lora":
        apply_lora(model, r=16, alpha=32)
    loaded = device_memory_mb(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.0)
    phases: dict[str, float] = {}  # the peak so far after each phase of the first step shows who sets the peak
    t0 = time.time()
    for i in range(steps + 1):
        batch = {k: v.to(device) for k, v in next(stream).items()}
        logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask")}, use_cache=False).logits
        loss = masked_lm_loss(logits, batch["labels"])
        phases.setdefault("forward_mb", round(device_memory_mb(device), 1))
        loss.backward()
        phases.setdefault("backward_mb", round(device_memory_mb(device), 1))
        opt.step()
        phases.setdefault("opt_step_mb", round(device_memory_mb(device), 1))
        opt.zero_grad(set_to_none=True)
        del logits, loss
        synchronize(device)
        if i == 0:  # the first step allocates optimizer state and, on MPS, compiles kernels
            t0 = time.time()
    trainable, total = count_parameters(model)
    return {"mode": mode, "hardware": hardware_label(device), "batch_size": batch_size, "trainable_params": trainable,
            "total_params": total, "loaded_mb": round(loaded, 1), **phases, "peak_mb": round(device_memory_mb(device), 1),
            "sec_per_step": round((time.time() - t0) / steps, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--worker", choices=["full", "lora"], default=None)
    ap.add_argument("--out", default="bench_memory.json")
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.steps, args.batch_size)))
        return

    rows = []
    for mode in ("full", "lora"):
        cmd = [sys.executable, __file__, "--worker", mode, "--steps", str(args.steps), "--batch-size", str(args.batch_size)]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        rows.append(json.loads(out.strip().splitlines()[-1]))
    what = "peak CUDA memory allocated" if "NVIDIA" in rows[0]["hardware"] else "peak of the training process"
    print(f"hardware: {rows[0]['hardware']}, batch {args.batch_size}, memory = {what}")
    print("MiB columns: peak so far after loading, then after the first forward, backward and optimizer step")
    print(f"{'mode':<8}{'trainable':>14}{'loaded':>9}{'forward':>9}{'backward':>10}{'opt step':>10}{'peak':>8}{'sec/step':>10}")
    for r in rows:
        print(f"{r['mode']:<8}{r['trainable_params']:>14,}{r['loaded_mb']:>9.0f}{r['forward_mb']:>9.0f}{r['backward_mb']:>10.0f}"
              f"{r['opt_step_mb']:>10.0f}{r['peak_mb']:>8.0f}{r['sec_per_step']:>10.2f}")
    save_json({"hardware": rows[0]["hardware"], "steps": args.steps, "rows": rows}, args.out)


if __name__ == "__main__":
    main()
