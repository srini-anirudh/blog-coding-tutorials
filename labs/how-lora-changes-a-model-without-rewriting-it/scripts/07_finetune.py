"""Full fine-tuning vs LoRA vs DoRA vs QLoRA on a real model (post: "PEFT versus full fine-tuning", "DoRA",
"QLoRA", Fig. 8 and 10). Small scale: SmolLM2-135M, a few million tokens.

Target domain: WikiText-2 with every word spelled backwards, a "language" the base model has never seen.
Source domain: plain WikiText-2, to measure forgetting.

For each run it records target and source validation loss before and after, trainable parameters, optimizer state,
peak memory and step time. It also records the post's two weight-space diagnostics:
DoRA's magnitude/direction changes (Delta M, Delta D) on q_proj and v_proj at checkpoints, and the number of
intruder dimensions in each adapted matrix at the end.

    python scripts/07_finetune.py --device cuda                 # all configs, one after another
    python scripts/07_finetune.py --configs lora8 --steps 20    # a short CPU check
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loralab.analysis import intruder_count, magnitude_direction, pearson
from loralab.lora import ALL_LINEAR, LoRALinear, apply_lora, merge_all
from loralab.quant import quantize_model_linears
from loralab.report import RESULTS, Report, table

CONFIGS = {  # name: (method, rank, learning rate)
    "full": ("full", None, 5e-5),
    "lora8": ("lora", 8, 5e-4),
    "lora64": ("lora", 64, 5e-4),
    "lora256": ("lora", 256, 5e-4),
    "dora8": ("dora", 8, 5e-4),
    "qlora8": ("qlora", 8, 5e-4),
}
TRACKED = ("q_proj", "v_proj")


def weights_of(model, kinds=ALL_LINEAR) -> dict[str, torch.Tensor]:
    """The effective (merged) weight of every linear layer whose name ends in one of `kinds`."""
    out = {}
    for name, mod in model.named_modules():
        if name.split(".")[-1] in kinds:
            out[name] = (mod.merged_weight() if isinstance(mod, LoRALinear) else mod.weight).detach().float().clone()
    return out


def build(model, method, rank):
    if method == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return
    if method == "qlora":
        quantize_model_linears(model, ALL_LINEAR)
    apply_lora(model, r=rank, alpha=2 * rank, dora=(method == "dora"))


def run(model, method, rank, lr, train_rows, evals, steps, batch, device, ckpt_every=50, seed=0):
    """Train one config. evals: {name: rows}. Returns a dict of measurements."""
    from loralab.data import mean_loss
    build(model, method, rank)
    model.to(device)
    w0 = weights_of(model)
    before = {k: mean_loss(model, v, device=device) for k, v in evals.items()}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    g = torch.Generator().manual_seed(seed)
    trace, losses = [], []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        x = train_rows[torch.randint(0, len(train_rows), (batch,), generator=g)].to(device)
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % ckpt_every == 0 or step == steps:
            w = weights_of(model, TRACKED)
            for name, wt in w.items():
                dm, dd = magnitude_direction(w0[name], wt)
                trace.append({"step": step, "layer": name, "dM": dm, "dD": dd})
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    after = {k: mean_loss(model, v, device=device) for k, v in evals.items()}
    state_bytes = sum(t.numel() * t.element_size() for s in opt.state.values() for t in s.values()
                      if torch.is_tensor(t) and t.dim() > 0)
    w_end = weights_of(model)
    intruders = {}
    for name in w_end:
        kind = name.split(".")[-1]
        intruders.setdefault(kind, []).append(intruder_count(w0[name], w_end[name]))
    out = {"method": method, "rank": rank, "lr": lr, "steps": steps, "batch": batch,
           "tokens": steps * batch * train_rows.shape[1],
           "trainable": sum(p.numel() for p in params), "adam_state_bytes": state_bytes,
           "step_time_s": elapsed / steps,
           "peak_mem_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
           "before": before, "after": after, "train_loss_last50": sum(losses[-50:]) / len(losses[-50:]),
           "intruders_mean_per_matrix": {k: sum(v) / len(v) for k, v in intruders.items()},
           "dM_dD_trace": trace, "dM_dD_corr": pearson([t["dM"] for t in trace], [t["dD"] for t in trace])}
    if method in ("lora", "dora"):
        key = next(iter(evals))
        merge_all(model)
        out["after_merge_" + key] = mean_loss(model, evals[key], device=device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--eval-rows", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--name", default="07_finetune")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from loralab.data import MODEL_ID, chunks, reverse_words, wikitext
    rep = Report(args.name, args.device)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    train_rows = chunks(tok, reverse_words(wikitext("train")), args.seq)
    evals = {"target (reversed words)": chunks(tok, reverse_words(wikitext("validation")), args.seq, args.eval_rows),
             "source (plain WikiText)": chunks(tok, wikitext("validation"), args.seq, args.eval_rows)}
    rep(f"{MODEL_ID}; train on {len(train_rows)} rows of {args.seq} reversed-word WikiText-2 tokens; "
        f"{args.steps} steps x batch {args.batch} = {args.steps * args.batch * args.seq / 1e6:.2f} M tokens; "
        f"eval {args.eval_rows} rows per domain; AdamW, no weight decay, linear warmup over 5% of steps, clip 1.0, fp32")
    results = {}
    for name in args.configs.split(","):
        method, rank, lr = CONFIGS[name]
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
        results[name] = run(model, method, rank, lr, train_rows, evals, args.steps, args.batch, args.device)
        r = results[name]
        rep(f"{name}: target {r['before']['target (reversed words)']:.3f} -> {r['after']['target (reversed words)']:.3f}, "
            f"source {r['before']['source (plain WikiText)']:.3f} -> {r['after']['source (plain WikiText)']:.3f}, "
            f"{r['step_time_s'] * 1e3:.0f} ms/step")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    rows = []
    for name, r in results.items():
        tb, ta = r["before"]["target (reversed words)"], r["after"]["target (reversed words)"]
        sb, sa = r["before"]["source (plain WikiText)"], r["after"]["source (plain WikiText)"]
        mem = f"{r['peak_mem_bytes'] / 1e9:.2f}" if r["peak_mem_bytes"] else "n/a"
        rows.append([name, f"{r['trainable'] / 1e6:.2f} M", f"{r['adam_state_bytes'] / 1e6:.1f}", mem,
                     f"{r['step_time_s'] * 1e3:.0f}", f"{tb:.3f}", f"{ta:.3f}", f"{sb:.3f}", f"{sa:.3f}",
                     f"{sa - sb:+.3f}"])
    rep("\nlosses are mean next-token cross entropy in nats")
    for line in table(rows, ["config", "trainable", "Adam MB", "peak GB", "ms/step", "target before", "target after",
                             "source before", "source after", "forgetting"]):
        rep(line)
    rep("\nweight-space diagnostics")
    rows = []
    for name, r in results.items():
        intr = r["intruders_mean_per_matrix"]
        last = [t for t in r["dM_dD_trace"] if t["step"] == r["steps"]]
        rows.append([name, f"{r['dM_dD_corr']:+.2f}", f"{sum(t['dM'] for t in last) / len(last):.2e}",
                     f"{sum(t['dD'] for t in last) / len(last):.2e}",
                     f"{sum(intr.values()) / len(intr):.2f}", f"{intr.get('q_proj', 0):.1f}",
                     f"{intr.get('down_proj', 0):.1f}"])
    for line in table(rows, ["config", "corr(dM, dD) q,v", "final dM", "final dD", "intruders / matrix (top 10)",
                             "q_proj", "down_proj"]):
        rep(line)
    rep("corr is Pearson over (layer, checkpoint) points for q_proj and v_proj, every 50 steps. "
        "Intruder: top-10 left singular vector with max |cos| < 0.5 against every pretrained singular vector.")
    for name, r in results.items():
        if any(k.startswith("after_merge") for k in r):
            k = next(k for k in r if k.startswith("after_merge"))
            rep(f"{name}: target loss unmerged {r['after']['target (reversed words)']:.5f}, merged {r[k]:.5f}")
    rep.data["config"] = vars(args)
    rep.data["runs"] = results
    rep.save()


if __name__ == "__main__":
    main()
