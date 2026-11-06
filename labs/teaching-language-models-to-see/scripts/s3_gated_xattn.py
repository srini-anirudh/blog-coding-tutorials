"""Section "Flamingo: keep the language model frozen and add new layers", and "Learning rates, warm-up and
initialization": a zero-initialised tanh gate is a no-op at step 0; a random MLP prefix is not.

    python labs/teaching-language-models-to-see/scripts/s3_gated_xattn.py [--steps 150]

1. At step 0, compare the caption-token predictions of the plain language model with
   (a) LLaVA-style: 49 image tokens from a random MLP connector put in front of the text, and
   (b) Flamingo-style: gated cross-attention layers inserted every 6 layers, gates at 0.
2. Train only the gated layers on C7's shape captions and watch the gates open.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seelab.common import Report, pick_device  # noqa: E402
from seelab.gated import FlamingoLite  # noqa: E402
from seelab.vlm import EVAL_SEED, TRAIN_SEED, build, lr_at  # noqa: E402
from tinyvlm.data import make_example, score  # noqa: E402


def text_only(messages: list[dict]) -> list[dict]:
    """Drop the image entry: the Flamingo model sees the picture through cross-attention, not as tokens."""
    user = {"role": "user", "content": [c for c in messages[0]["content"] if c["type"] != "image"]}
    return [user, *messages[1:]]


def batches(processor, examples, with_image: bool):
    if with_image:
        b = processor.collate([processor.build_example(e["messages"], e["images"]) for e in examples])
    else:
        b = processor.collate([processor.build_example(text_only(e["messages"])) for e in examples])
        b["pixel_values"] = processor.image_processor([e["images"][0] for e in examples])["pixel_values"]
    return b


def reply_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Log-softmax at every position that predicts a reply token, in order. (positions x V)"""
    pos = labels[:, 1:] != -100
    return F.log_softmax(logits[:, :-1][pos].float(), dim=-1), labels[:, 1:][pos]


@torch.no_grad()
def flamingo_captions(fl, processor, n, device):
    examples = [make_example(i, seed=EVAL_SEED) for i in range(n)]
    prompts = [processor.apply_chat_template(text_only(e["messages"])[:1], add_generation_prompt=True) for e in examples]
    b = processor(prompts)
    px = processor.image_processor([e["images"][0] for e in examples])["pixel_values"]
    new = fl.generate(b["input_ids"].to(device), b["attention_mask"].to(device), px.to(device),
                      max_new_tokens=12, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
    preds = [p.strip() for p in processor.batch_decode(new)]
    return preds, score(preds, [e["target"] for e in examples])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--checkpoints", default="10,25,50,100,150")
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    rep = Report("s3_gated_xattn", device)
    vlm, processor = build("mlp")
    vlm.to(device).eval()
    lm = vlm.lm
    ev = [make_example(20_000 + i, seed=EVAL_SEED) for i in range(16)]
    txt, img = batches(processor, ev, False), batches(processor, ev, True)

    rep("## 1. Step 0: how much does adding the image change the caption predictions?  16 held-out pictures")
    with torch.no_grad():
        base_logits = lm(input_ids=txt["input_ids"].to(device), attention_mask=txt["attention_mask"].to(device)).logits
        base_lp, target = reply_logprobs(base_logits, txt["labels"].to(device))
        llava_logits = vlm(input_ids=img["input_ids"].to(device), pixel_values=img["pixel_values"].to(device),
                           attention_mask=img["attention_mask"].to(device)).logits
        llava_lp, target2 = reply_logprobs(llava_logits, img["labels"].to(device))
    assert torch.equal(target, target2)
    nll = lambda lp: -lp.gather(1, target[:, None]).mean().item()  # noqa: E731
    kl = lambda lp: F.kl_div(lp, base_lp, log_target=True, reduction="batchmean").item()  # KL(base || lp)  # noqa: E731
    agree = lambda lp: (lp.argmax(-1) == base_lp.argmax(-1)).float().mean().item()  # noqa: E731

    fl = FlamingoLite(vlm.vision, lm).to(device)
    n_train = sum(p.numel() for p in fl.trainable())
    with torch.no_grad():
        fl_logits = fl(txt["input_ids"].to(device), txt["attention_mask"].to(device),
                       pixel_values=txt["pixel_values"].to(device)).logits
        fl_lp, _ = reply_logprobs(fl_logits, txt["labels"].to(device))
    rows = {
        "language model, no image": {"nll": nll(base_lp), "kl": 0.0, "top1_agree": 1.0, "max_abs_logit_diff": 0.0},
        "LLaVA-style, random MLP prefix": {"nll": nll(llava_lp), "kl": kl(llava_lp), "top1_agree": agree(llava_lp)},
        "Flamingo-style, gates at 0": {"nll": nll(fl_lp), "kl": kl(fl_lp), "top1_agree": agree(fl_lp),
                                       "max_abs_logit_diff": (fl_logits - base_logits).abs().max().item()},
    }
    rep(f"{'model':<34} {'reply NLL':>9} {'KL to LM':>9} {'same top-1':>10}")
    for k, r in rows.items():
        rep(f"{k:<34} {r['nll']:9.3f} {r['kl']:9.4f} {r['top1_agree']:10.2f}")
    rep(f"Flamingo-style at init: max |logit difference| to the plain language model = {rows['Flamingo-style, gates at 0']['max_abs_logit_diff']:.1e}")
    rep(f"{len(fl.gated)} gated blocks (before layers {', '.join(str(i) for i in range(0, 30, 6))}), {n_train:,} trainable parameters")

    rep(f"\n## 2. Train only the gated blocks: {args.steps} steps x {args.batch_size} pictures, AdamW lr {args.lr}")
    opt = torch.optim.AdamW(fl.trainable(), lr=args.lr, weight_decay=0.0)
    checkpoints = {int(c) for c in args.checkpoints.split(",")}
    history, seconds = [], 0.0
    fl.train()
    for step in range(args.steps):
        t0 = time.time()
        ex = [make_example(step * args.batch_size + i, seed=TRAIN_SEED) for i in range(args.batch_size)]
        b = batches(processor, ex, False)
        loss = fl(b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device),
                  b["pixel_values"].to(device)).loss
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, args.steps, args.lr, 10)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:  # with every gate at 0, only the gates themselves receive gradient
            g_alpha = sum(g.xattn.alpha_attn.grad.abs().item() + g.xattn.alpha_ff.grad.abs().item() for g in fl.gated)
            g_rest = sum(p.grad.abs().sum().item() for g in fl.gated for n, p in g.xattn.named_parameters()
                         if not n.startswith("alpha") and p.grad is not None)
            rep(f"step 0 gradients: |gates| {g_alpha:.3e}, |all other new weights| {g_rest:.3e}")
        torch.nn.utils.clip_grad_norm_(fl.trainable(), 1.0)
        opt.step()
        seconds += time.time() - t0
        if step + 1 in checkpoints:
            gates = fl.gates()
            fl.eval()
            with torch.no_grad():  # how much do the gated blocks actually add to the residual stream?
                fl(txt["input_ids"].to(device), txt["attention_mask"].to(device), pixel_values=txt["pixel_values"].to(device))
            contrib = fl.contributions()
            preds, sc = flamingo_captions(fl, processor, args.eval_n, device)
            fl.train()
            history.append({"step": step + 1, "loss": loss.item(), "gates": gates, "added_over_residual": contrib,
                            "all_correct": sc["all"], "sample": preds[0]})
            rep(f"step {step + 1:>4} loss {loss.item():.3f}  tanh(gate) attn {[round(a, 3) for a, _ in gates]}  "
                f"ffw {[round(f, 3) for _, f in gates]}  |added|/|residual| {[round(c, 3) for c in contrib]}  "
                f"captions all-correct {sc['all']:.2f}  e.g. {preds[0]!r}")
    rep(f"training time {seconds:.0f} s")
    rep("Compare: s2 trains a 2.1M-parameter MLP prefix connector with the same data, steps and batch size.")
    rep.save({"step0": rows, "trainable_params": n_train, "history": history, "args": vars(args), "train_seconds": seconds})


if __name__ == "__main__":
    main()
