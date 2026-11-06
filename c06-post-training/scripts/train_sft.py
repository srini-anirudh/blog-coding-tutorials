"""Stage 1: SFT SmolLM2-135M on arithmetic conversations.

    python scripts/train_sft.py --mode full            # full fine-tuning, padded batches
    python scripts/train_sft.py --mode lora            # LoRA r=16 on every linear layer
    python scripts/train_sft.py --mode full --pack     # packed batches
    python scripts/train_sft.py --init random          # same recipe from random weights (why we need a base)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.chat import load_tokenizer  # noqa: E402
from posttrain.evaluate import evaluate  # noqa: E402
from posttrain.lora import apply_lora, lora_state_dict, merge_lora  # noqa: E402
from posttrain.models import load_policy, random_policy  # noqa: E402
from posttrain.sft import SFTConfig, build_examples, train_sft  # noqa: E402
from posttrain.tasks import split_problems  # noqa: E402
from posttrain.utils import BASE_MODEL, count_parameters, dir_size_mb, hardware_label, pick_device, polite_threads, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lora"], default="full")
    ap.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    polite_threads()
    torch.manual_seed(0)
    device = pick_device()
    tok = load_tokenizer()
    train, test = split_problems(args.n_train, args.n_test, seed=0, follow_up_frac=0.3)
    examples = build_examples(tok, train)

    model = load_policy(BASE_MODEL, device) if args.init == "pretrained" else random_policy(BASE_MODEL, device)
    if args.mode == "lora":
        apply_lora(model, r=args.rank, alpha=2 * args.rank)
    lr = args.lr or (1e-3 if args.mode == "lora" else 5e-5)
    trainable, total = count_parameters(model)
    print(f"hardware: {hardware_label(device)} trainable={trainable:,} total={total:,} examples={len(examples)}")

    before = evaluate(model, tok, test) if args.init == "pretrained" else None
    print("before", before)
    cfg = SFTConfig(steps=args.steps, batch_size=args.batch_size, lr=lr, pack=args.pack, max_len=256,
                    eval_fn=lambda m: evaluate(m, tok, test[:100]), eval_every=50)
    t0 = time.time()
    history = train_sft(model, examples, tok.pad_token_id, cfg, device)
    train_sec = time.time() - t0
    after = evaluate(model, tok, test)
    print("after", after)

    tag = args.tag or f"sft-{args.mode}{'-packed' if args.pack else ''}{'-random' if args.init == 'random' else ''}"
    out = ROOT / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "lora":
        adapter = out / "adapter"
        adapter.mkdir(exist_ok=True)
        save_file(lora_state_dict(model), str(adapter / "adapter.safetensors"))
        ckpt_mb = dir_size_mb(adapter)
        merge_lora(model)  # downstream stages load a plain model
        model.save_pretrained(out / "merged")
    else:
        model.save_pretrained(out)
        ckpt_mb = dir_size_mb(out)
    tok.save_pretrained(out)

    save_json({
        "tag": tag, "hardware": hardware_label(device), "mode": args.mode, "init": args.init, "pack": args.pack,
        "lr": lr, "steps": args.steps, "batch_size": args.batch_size,
        "trainable_params": trainable, "total_params": total, "checkpoint_mb": round(ckpt_mb, 1),
        "peak_mem_mb": max(h["mem_mb"] for h in history), "train_sec": round(train_sec, 1),
        "before": before, "after": after, "history": history,
    }, f"{tag}.json")


if __name__ == "__main__":
    main()
