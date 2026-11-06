"""Sample stories from a checkpoint.

    python c05-pretraining/scripts/sample.py --ckpt c05-pretraining/runs/tinystories/ckpt_000800.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pretrain.activation_ckpt import apply_activation_checkpointing  # noqa: E402
from pretrain.config import TrainConfig  # noqa: E402
from pretrain.model import build_model, sample  # noqa: E402
from pretrain.tokenizer import load_file  # noqa: E402
from pretrain.utils import pick_device, resolve  # noqa: E402

PROMPTS = ["Once upon a time", "Lily found a", "The little dog was sad because"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="checkpoint path; omit for an untrained model with --config")
    ap.add_argument("--config", default="configs/tinystories.yaml")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        cfg = TrainConfig.from_dict(state["config"])
    else:
        state, cfg = None, TrainConfig.from_yaml(resolve(args.config))
    model = build_model(cfg.model, cfg.model_type)
    if cfg.activation_checkpointing:
        apply_activation_checkpointing(model)
    if state:
        model.load_state_dict(state["model"])
    model.to(device).eval()
    tok = load_file(resolve(cfg.data_dir) / f"bpe{cfg.model.vocab_size}.json")
    g = torch.Generator().manual_seed(0)
    print(f"step {state['step'] if state else 0}, device {device}, temperature {args.temperature}, top-k {args.top_k}")
    for p in PROMPTS:
        ids = tok.encode(p)
        out = sample(model, ids, args.max_new, tok.eos_id, args.temperature, args.top_k, cfg.seq_len, g)
        stopped = out[-1] == tok.eos_id
        print(f"\n[{p}] {tok.decode(out)}\n  ({len(out)} tokens, {'stopped at EOS' if stopped else 'hit the limit'})")


if __name__ == "__main__":
    main()
