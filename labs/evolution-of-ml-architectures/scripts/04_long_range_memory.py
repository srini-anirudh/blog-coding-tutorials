"""Train five mixers on one long-range task: remember the token after MARK across n filler tokens.

This is the post's "same sentence through four architectures" lab, made measurable, plus an LTI SSM.
For each model and distance: final accuracy on fresh sequences, parameters, and training time per step.
For the selective SSM it also reads out the learned step size Delta at the key versus at the filler,
which is the post's claim that the gate "opens on keys and nearly closes on the filler tokens".

    python labs/evolution-of-ml-architectures/scripts/04_long_range_memory.py            # full run
    python labs/evolution-of-ml-architectures/scripts/04_long_range_memory.py --quick    # smoke test
"""

import _setup  # noqa: F401

import argparse
import json
import time

import torch
import torch.nn.functional as F

from archlab.counts import n_params
from archlab.env import RESULTS, start
from archlab.seqmodels import SeqModel, key_recall_batch

p = argparse.ArgumentParser()
p.add_argument("--fillers", type=int, nargs="+", default=[16, 64, 128])
p.add_argument("--models", nargs="+", default=["rnn", "lstm", "attention", "lti_ssm", "selective_ssm"])
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--seeds", type=int, nargs="+", default=[0])
p.add_argument("--quick", action="store_true")
p.add_argument("--device", default="cpu", help="cpu or cuda")
p.add_argument("--tag", default="", help="suffix for the results files, to run distances in parallel")
args = p.parse_args()
if args.quick:
    args.fillers, args.steps = [8], 60
NAME = "04_long_range_memory" + ("_quick" if args.quick else args.tag)
start(NAME, device=args.device)
DEV = torch.device(args.device)

N_KEYS = 16


def train(kind, n_filler, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    model = SeqModel(kind, vocab=N_KEYS + 2, n_classes=N_KEYS, d=64, max_len=n_filler + 3).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.steps, pct_start=0.1)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        x, y = (t.to(DEV) for t in key_recall_batch(args.batch, n_filler, N_KEYS, gen))
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    ms_per_step = 1000 * (time.perf_counter() - t0) / args.steps
    model.eval()
    eval_gen = torch.Generator().manual_seed(99)
    correct = 0
    with torch.no_grad():
        for _ in range(8):
            x, y = (t.to(DEV) for t in key_recall_batch(256, n_filler, N_KEYS, eval_gen))
            correct += (model(x).argmax(-1) == y).sum().item()
    row = dict(model=kind, n_filler=n_filler, seed=seed, params=n_params(model), acc=correct / 2048,
               final_loss=round(loss.item(), 4), ms_per_step=round(ms_per_step, 1))
    if kind == "selective_ssm":
        dt = model.mixers[0].last_dt.mean(-1)  # (B, N): mean Delta over channels, layer 1, last eval batch
        row["delta_at_key"] = round(dt[:, 1].mean().item(), 4)
        row["delta_at_filler"] = round(dt[:, 2:-1].mean().item(), 4)
    return row


rows = []
print(f"task: [MARK, key, n filler keys, QUERY] -> key; {N_KEYS} keys, chance = {1 / N_KEYS:.3f}")
print(f"training: {args.steps} steps, batch {args.batch}, AdamW lr 2e-3 one-cycle, clip 1.0, width 64, 2 layers")
print()
print(f"{'model':>14} {'filler':>6} {'seed':>4} {'params':>7} {'acc':>6} {'loss':>7} {'ms/step':>8}  gate")
for n_filler in args.fillers:
    for kind in args.models:
        for seed in args.seeds:
            r = train(kind, n_filler, seed)
            rows.append(r)
            gate = f"Delta key {r['delta_at_key']:.3f} vs filler {r['delta_at_filler']:.3f}" if "delta_at_key" in r else ""
            print(f"{kind:>14} {n_filler:>6} {seed:>4} {r['params']:>7} {r['acc']:>6.3f} {r['final_loss']:>7.3f} "
                  f"{r['ms_per_step']:>8.1f}  {gate}", flush=True)

if not args.quick:
    (RESULTS / f"{NAME}.json").write_text(json.dumps(rows, indent=1))
