"""Sample from a saved checkpoint, or compare temperature, top-k and top-p side by side.

    python c04-llm/scripts/generate.py --prompt "ROMEO:" --temperature 0.8 --top-k 40
    python c04-llm/scripts/generate.py --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm import ByteTokenizer, load_checkpoint  # noqa: E402
from tinyllm.utils import pick_device  # noqa: E402

SETTINGS = [
    ("greedy", dict(temperature=0.0)),
    ("temperature 0.5", dict(temperature=0.5)),
    ("temperature 1.0", dict(temperature=1.0)),
    ("temperature 1.5", dict(temperature=1.5)),
    ("temperature 1.0, top_k 5", dict(temperature=1.0, top_k=5)),
    ("temperature 1.0, top_p 0.5", dict(temperature=1.0, top_p=0.5)),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "runs" / "tiny-shakespeare"))
    p.add_argument("--prompt", default="ROMEO:\n")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--compare", action="store_true", help="print one sample for each of several settings")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if not (Path(args.checkpoint) / "model.safetensors").exists():
        sys.exit("run scripts/train_sanity.py first: this script needs a trained checkpoint")
    device = pick_device(args.device)
    model = load_checkpoint(args.checkpoint, device).eval()
    tok = ByteTokenizer()
    ids = torch.tensor([tok.encode(args.prompt)], device=device)
    common = dict(max_new_tokens=args.max_new_tokens, seed=args.seed, use_cache=not args.no_cache)

    if args.compare:
        for label, setting in SETTINGS:
            text = tok.decode(model.generate(ids, **common, **setting)[0].tolist())
            print(f"[{label}]\n{text}\n")
        return
    out = model.generate(ids, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, **common)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
