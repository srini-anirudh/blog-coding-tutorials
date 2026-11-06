"""Section "Why freeze anything" and the Nuance on the alignment stage: does a connector-only stage protect the
language model when it is unfrozen?

    python labs/teaching-language-models-to-see/scripts/s7_alignment_stage.py [--steps 60]   # needs s2's connector

Two runs of the same joint stage (connector lr 1e-3, language model lr --lm-lr, vision frozen):
    no alignment   the connector starts from random weights
    aligned first  the connector starts from s2's connector-only stage (150 steps)
We measure the language model's perplexity on held-out WikiText-2 text (no image), which is what a noisy
early gradient would damage, and caption accuracy. C8's forgetting check and text replay come from C8's code.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seelab.common import REPO, RUNS, Report, pick_device  # noqa: E402
from seelab.vlm import batch_of, build, caption_eval, lr_at, to  # noqa: E402

sys.path.insert(0, str(REPO / "c08-vlm-training"))
from vlmtrain.replay import load_passages, perplexity  # noqa: E402


def run(name: str, init: Path | None, args, device, passages, rep: Report) -> list[dict]:
    model, processor = build("mlp")
    if init is not None:
        model.projector.load_state_dict(load_file(init))
    model.set_trainable(vision=False, projector=True, lm=True)
    model.to(device)
    opt = torch.optim.AdamW([{"params": model.projector.parameters(), "lr": args.lr, "peak": args.lr},
                             {"params": model.lm.parameters(), "lr": args.lm_lr, "peak": args.lm_lr}], weight_decay=0.0)
    checkpoints = {int(c) for c in args.checkpoints.split(",")}
    rows, seconds = [], 0.0

    def measure(step):
        model.eval()
        ppl = perplexity(model.lm, processor.tokenizer, passages, device)
        _, sc = caption_eval(model, processor, args.eval_n, device)
        rows.append({"step": step, "ppl": ppl, "all_correct": sc["all"], "train_seconds": seconds})
        rep(f"{name:<14} step {step:>3}  WikiText-2 perplexity {ppl:7.2f}  captions all-correct {sc['all']:.2f}")
        model.train()

    measure(0)
    model.train()
    for step in range(args.steps):
        t0 = time.time()
        batch, _ = batch_of(processor, 50_000 + step * args.batch_size, args.batch_size)  # not s2's pictures
        loss = model(**to(batch, device)).loss
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, args.steps, gr["peak"], args.warmup)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step < 3:
            gp = torch.nn.utils.get_total_norm([p.grad for p in model.lm.parameters() if p.grad is not None]).item()
            rep(f"{name:<14} step {step:>3}  loss {loss.item():.3f}  language-model gradient norm {gp:.2f}")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        seconds += time.time() - t0
        if step + 1 in checkpoints:
            measure(step + 1)
    del model
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lm-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--checkpoints", default="5,15,30,60")
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--passages", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--connector", default="runs/s2_connector/projector.safetensors", help="relative to the lab folder")
    ap.add_argument("--name", default="s7_alignment_stage")
    args = ap.parse_args()
    device = pick_device(args.device)
    rep = Report(args.name, device)
    ckpt = RUNS.parent / args.connector
    if not ckpt.exists():
        sys.exit("missing runs/s2_connector/projector.safetensors: run scripts/s2_connector_gap.py first")
    passages = load_passages("test", limit=args.passages)
    rep(f"joint stage: {args.steps} steps x {args.batch_size} pictures; connector lr {args.lr}, LM lr {args.lm_lr}, "
        f"warmup {args.warmup}, grad clip 1.0; perplexity on {len(passages)} WikiText-2 test passages (128 tokens)")
    out = {"args": vars(args)}
    for name, init in (("no alignment", None), ("aligned first", ckpt)):
        out[name] = run(name, init, args, device, passages, rep)
    a, b = out["no alignment"][-1], out["aligned first"][-1]
    base = out["no alignment"][0]["ppl"]
    rep(f"\nafter {args.steps} joint steps: perplexity {base:.2f} -> {a['ppl']:.2f} (no alignment) and "
        f"{b['ppl']:.2f} (aligned first); captions {a['all_correct']:.2f} and {b['all_correct']:.2f}")
    rep("The aligned run also had s2's 150 connector-only steps before this stage; the comparison is about")
    rep("what the language model goes through, not about equal compute.")
    rep.save(out)


if __name__ == "__main__":
    main()
