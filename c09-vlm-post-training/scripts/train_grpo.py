"""Visual GRPO on SmolVLM-256M with verifiable rewards from synthetic scenes.

    python scripts/train_grpo.py --task count --reward exact --freeze llm --steps 60
    python scripts/train_grpo.py --task caption --reward color_recall --freeze llm   # the hackable reward
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlmalign.device import pick_device, seed_everything, sync  # noqa: E402
from vlmalign.evaluate import text_count_accuracy, vqa_accuracy, vqa_items  # noqa: E402
from vlmalign.freeze import configure_trainable  # noqa: E402
from vlmalign.grpo import group_advantages, grpo_loss, policy_token_logprobs, rollout_batch, sample_group  # noqa: E402
from vlmalign.lora import lora_delta_norms  # noqa: E402
from vlmalign.model import encode_prompt, generate, load_model, load_processor  # noqa: E402
from vlmalign.rl_tasks import REWARDS, TASKS, caption_examples, caption_stats  # noqa: E402
from vlmalign.scenes import render  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), default="count")
    p.add_argument("--reward", choices=list(REWARDS), default="exact")
    p.add_argument("--freeze", choices=["llm", "projector+llm", "all"], default="llm")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--prompts", type=int, default=4, help="images per step (P)")
    p.add_argument("--group", type=int, default=8, help="rollouts per image (G)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", choices=["none", "after", "both"], default="after")
    p.add_argument("--name", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, proc, task: str, seed: int = 12345) -> dict:
    if task == "count":
        count_items = vqa_items(100, seed, kinds=("count",))
        other_items = vqa_items(30, seed + 1, kinds=("color", "object", "spatial", "ocr"))
        return {"count": vqa_accuracy(model, proc, count_items)["count"],
                "other": vqa_accuracy(model, proc, other_items),
                "text_count": text_count_accuracy(model, proc, count_items[:60])}
    stats = []
    for ex in caption_examples(50, seed):
        text = generate(model, proc, render(ex.scene), ex.prompt, max_new_tokens=24)[0]
        stats.append(caption_stats(text, ex))
    return {k: sum(s[k] for s in stats) / len(stats) for k in stats[0]}


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(2)
    seed_everything(args.seed)
    device = pick_device()
    name = args.name or f"grpo_{args.task}_{args.reward}_{args.freeze.replace('+', '_')}"
    proc, model = load_processor(), load_model(device=device)
    counts = configure_trainable(model, args.freeze)
    print(f"[{name}] device={device} trainable={counts}")
    pad = proc.tokenizer.pad_token_id
    strict = "exact" if args.task == "count" else "color_f1"   # the reward we actually care about

    result = {"args": vars(args), "trainable": counts, "log": []}
    if args.eval == "both":
        result["before"] = evaluate(model, proc, args.task)
        print("before", result["before"])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    examples = TASKS[args.task](args.steps * args.prompts, seed=args.seed + 1)
    t0 = time.time()
    for step in range(args.steps):
        batch_ex = examples[step * args.prompts: (step + 1) * args.prompts]
        # 1) rollouts: G samples per image from the current policy
        model.eval()
        groups = [sample_group(model, proc, encode_prompt(proc, render(ex.scene), ex.prompt), args.group,
                               max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                  for ex in batch_ex]
        # 2) verifiable rewards, read off the scene
        rewards = torch.tensor([[REWARDS[args.reward](t, ex) for t in g.texts] for g, ex in zip(groups, batch_ex)])
        strict_r = torch.tensor([[REWARDS[strict](t, ex) for t in g.texts] for g, ex in zip(groups, batch_ex)])
        # 3) group-relative advantages: (P, G) -> (P*G,)
        adv = group_advantages(rewards).flatten().to(device)
        # 4) one on-policy update
        model.train()
        batch = rollout_batch(groups, pad).to(device)
        logp, mask = policy_token_logprobs(model, batch)
        loss = grpo_loss(logp, logp.detach(), adv, mask.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sync(device)

        entry = {"step": step, "reward": rewards.mean().item(), "strict": strict_r.mean().item(),
                 "zero_std_groups": (rewards.std(1) == 0).float().mean().item(),
                 "length": torch.stack([g.lengths for g in groups]).float().mean().item(),
                 "loss": loss.item(), "grad_norm": gnorm.item(), "sample": groups[0].texts[:4],
                 "elapsed": time.time() - t0}
        if args.task == "caption":
            st = [caption_stats(t, ex) for g, ex in zip(groups, batch_ex) for t in g.texts]
            entry["n_colors"] = sum(s["n_colors"] for s in st) / len(st)
            entry["hallucinated"] = sum(s["hallucinated"] for s in st) / len(st)
        result["log"].append(entry)
        if step % 5 == 0 or step == args.steps - 1:
            extra = f" colors={entry['n_colors']:.2f} halluc={entry['hallucinated']:.2f}" if args.task == "caption" else ""
            print(f"step {step:3d} reward={entry['reward']:.3f} strict={entry['strict']:.3f} "
                  f"len={entry['length']:.1f} flat_groups={entry['zero_std_groups']:.2f}{extra} "
                  f"t={entry['elapsed']:.0f}s | {groups[0].texts[0]!r}")

    result["train_seconds"] = time.time() - t0
    result["delta_norm"] = {part: lora_delta_norms(model, prefix) for part, prefix in
                            [("llm", "model.text_model"), ("vision", "model.vision_model")]}
    if args.eval != "none":
        result["after"] = evaluate(model, proc, args.task)
        print("after", result["after"])
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1))
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    (ROOT / "runs").mkdir(exist_ok=True)
    torch.save(trainable, ROOT / "runs" / f"{name}.pt")
    return result


if __name__ == "__main__":
    main()
