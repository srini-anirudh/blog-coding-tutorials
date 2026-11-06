"""One GRPO group, printed: G sampled answers, their rewards and advantages.

    python scripts/show_rollouts.py --group 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlmalign.device import pick_device, seed_everything  # noqa: E402
from vlmalign.grpo import group_advantages, rollout_batch, sample_group  # noqa: E402
from vlmalign.logprobs import IGNORE  # noqa: E402
from vlmalign.model import encode_prompt, load_model, load_processor  # noqa: E402
from vlmalign.rl_tasks import REWARDS, count_examples  # noqa: E402
from vlmalign.scenes import render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--images", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(2)
    seed_everything(0)
    device = pick_device()
    proc, model = load_processor(), load_model(device=device).eval()

    dump = []
    for ex in count_examples(args.images, seed=2):
        g = sample_group(model, proc, encode_prompt(proc, render(ex.scene), ex.prompt), args.group, max_new_tokens=12)
        r = torch.tensor([[REWARDS["exact"](t, ex) for t in g.texts]])
        a = group_advantages(r)[0]
        print(f"\nprompt: {ex.prompt}   truth: {ex.answer}")
        print(f"{'rollout':>7s} {'tokens':>6s} {'reward':>6s} {'adv':>6s}  text")
        for i, t in enumerate(g.texts):
            print(f"{i:7d} {int(g.lengths[i]):6d} {r[0, i]:6.1f} {a[i]:+6.2f}  {t!r}")
        dump.append({"prompt": ex.prompt, "truth": ex.answer, "texts": g.texts, "rewards": r[0].tolist(),
                     "advantages": a.tolist(), "objects": [[o.shape, o.color, list(o.box)] for o in ex.scene.objects]})

    b = rollout_batch([g], proc.tokenizer.pad_token_id)
    print("\none group as a batch")
    for name in ("input_ids", "attention_mask", "labels", "image_index", "pixel_values"):
        t = getattr(b, name)
        print(f"{name:15s} {t.shape}")
    print(f"answer tokens per row: {(b.labels != IGNORE).sum(1).tolist()}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "show_rollouts.json").write_text(json.dumps(dump, indent=1))


if __name__ == "__main__":
    main()
