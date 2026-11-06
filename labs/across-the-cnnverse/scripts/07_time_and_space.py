"""Time and space: do MACs predict latency, and what fills memory during a training step?

1. Forward latency of the nine networks against their MAC counts.
2. Throughput of single operators at 56x56x128: standard, 1x1, grouped, depthwise. Depthwise has low
   arithmetic intensity (MACs per byte moved), so it runs far below peak.
3. Training memory: weights, gradients, Adam state, and the activations autograd saves for backward,
   measured with saved-tensor hooks. Then activation checkpointing on ResNet-50.
On CUDA, part 3 also measures peak allocated memory and step time for a real training step.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

from cnnverse import zoo
from cnnverse.profile import profile, saved_activation_bytes
from cnnverse.report import RESULTS, Report, set_threads


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def median_time(fn, dev, reps, warmup=3):
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def latency_table(R, dev, batch, reps, out):
    R(f"\n== Forward latency vs MACs, batch {batch}, fp32, eval mode, median of {reps} ==")
    R(f"{'network':>16} {'GMACs/img':>10} {'ms/batch':>9} {'achieved GMAC/s':>16} {'MAC ratio to R50':>17}"
      f" {'time ratio to R50':>18}")
    rows = {}
    for name, (build, inp, *_) in zoo.NINE.items():
        if name == "LeNet-5":
            continue
        macs = profile(build(), inp).macs
        model = build().to(dev).eval()
        x = torch.randn(batch, *inp, device=dev)
        with torch.no_grad():
            t = median_time(lambda: model(x), dev, reps)
        rows[name] = (macs, t)
        del model
    m50, t50 = rows["ResNet-50"]
    for name, (macs, t) in rows.items():
        R(f"{name:>16} {macs / 1e9:>10.3f} {t * 1e3:>9.2f} {macs * batch / t / 1e9:>16.1f} {macs / m50:>17.2f}"
          f" {t / t50:>18.2f}")
    mv, tv = rows["MobileNetV2"]
    R(f"MobileNetV2 has {m50 / mv:.1f}x fewer MACs than ResNet-50 and runs {t50 / tv:.1f}x faster here.")
    out[f"latency_b{batch}"] = {k: {"macs": m, "seconds": t} for k, (m, t) in rows.items()}


def operator_table(R, dev, batch, reps, out):
    c, hw = 128, 56
    R(f"\n== One operator at {hw}x{hw}, {c} -> {c} channels, batch {batch}, fp32 ==")
    R(f"{'operator':>20} {'MACs':>9} {'MAC/byte':>9} {'ms':>8} {'achieved GMAC/s':>16}")
    ops = {"standard 3x3": nn.Conv2d(c, c, 3, padding=1, bias=False),
           "1x1 pointwise": nn.Conv2d(c, c, 1, bias=False),
           "grouped 3x3, g=32": nn.Conv2d(c, c, 3, padding=1, groups=32, bias=False),
           "depthwise 3x3": nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)}
    x = torch.randn(batch, c, hw, hw, device=dev)
    res = {}
    for name, op in ops.items():
        op = op.to(dev)
        macs = batch * hw * hw * c * op.weight[0].numel()
        bytes_moved = 4 * (2 * x.numel() + op.weight.numel())  # read input, write output, read weights once
        with torch.no_grad():
            t = median_time(lambda: op(x), dev, reps)
        res[name] = {"macs": macs, "intensity": macs / bytes_moved, "seconds": t}
        R(f"{name:>20} {macs / 1e6:>8.0f}M {macs / bytes_moved:>9.1f} {t * 1e3:>8.3f} {macs / t / 1e9:>16.1f}")
    s, d = res["standard 3x3"], res["depthwise 3x3"]
    R(f"depthwise does {s['macs'] / d['macs']:.0f}x fewer MACs than standard 3x3 and runs {s['seconds'] / d['seconds']:.1f}x"
      f" faster.")
    out[f"operators_b{batch}"] = res


MEMORY_MODELS = ("VGG-16", "ResNet-50", "DenseNet-121", "MobileNetV2", "ConvNeXt-T")


def memory_table(R, out, batch=2, scale_to=32):
    R(f"\n== Training memory, fp32 + Adam. Saved activations measured at batch {batch}, shown per image and"
      f" x{scale_to} ==")
    R(f"{'network':>14} {'weights':>9} {'grads':>8} {'Adam m,v':>9} {'outputs/img':>12} {'saved/img':>10}"
      f" {'saved x32':>10} {'saved x32 / weights':>20}")
    res = {}
    for name in MEMORY_MODELS:
        build, inp, *_ = zoo.NINE[name]
        prof = profile(build(), inp)
        model = build().train()
        w = sum(p.numel() for p in model.parameters()) * 4
        saved = saved_activation_bytes(model, torch.randn(batch, *inp)) / batch
        outputs = prof.act_elems * 4
        res[name] = {"weights": w, "saved_per_image": saved, "layer_outputs_per_image": outputs}
        R(f"{name:>14} {w / 2**30:>7.2f}Gi {w / 2**30:>6.2f}Gi {2 * w / 2**30:>7.2f}Gi {outputs / 2**20:>9.0f}MiB"
          f" {saved / 2**20:>7.0f}MiB {saved * scale_to / 2**30:>8.2f}Gi {saved * scale_to / w:>20.1f}")
        del model
    R("outputs/img counts only conv, linear and pool outputs (the spec engine's figure); saved/img is what")
    R("autograd actually keeps: also BN inputs, ReLU outputs, concatenations and LayerNorm statistics.")
    out["memory"] = res

    R("\n== Activation checkpointing, ResNet-50, batch 2, saved bytes after forward ==")
    model = zoo.resnet50().train()
    for seg in (0, 2, 4, 8):
        x = torch.randn(2, 3, 224, 224, requires_grad=seg > 0)
        b = saved_activation_bytes(model, x) if seg == 0 else _saved_ckpt(model, x, seg)
        R(f"  segments {seg if seg else 'off':>3}: {b / 2 / 2**20:7.1f} MiB per image")
        out.setdefault("checkpoint_cpu", {})[seg] = b / 2
    R("checkpoint_sequential keeps only each segment's input, plus the last segment in full; the rest is")
    R("recomputed during backward. The peak during backward also holds one segment's recomputed activations.")


def _saved_ckpt(model, x, segments):
    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, t):
            return checkpoint_sequential(self.m, segments, t, use_reentrant=True)
    return saved_activation_bytes(Wrap(model), x)


def cuda_training_step(R, dev, out, batch=32):
    R(f"\n== CUDA training step, batch {batch}, fp32, Adam: peak allocated memory and step time ==")
    R(f"{'network':>14} {'checkpoint':>10} {'peak':>8} {'weights+grads+Adam':>19} {'step ms':>8}")
    res = {}
    for name in MEMORY_MODELS:
        for ckpt in (False, True):
            if ckpt and name not in ("VGG-16", "ResNet-50"):
                continue
            build, inp, *_ = zoo.NINE[name]
            model = build().to(dev).train()
            opt = torch.optim.Adam(model.parameters())
            x = torch.randn(batch, *inp, device=dev, requires_grad=ckpt)
            y = torch.randint(0, 1000, (batch,), device=dev)
            seq = model[0] if name == "VGG-16" else model

            def step():
                if ckpt and name == "VGG-16":
                    logits = model[1](checkpoint_sequential(seq, 4, x, use_reentrant=True))
                elif ckpt:
                    logits = checkpoint_sequential(model, 4, x, use_reentrant=True)
                else:
                    logits = model(x)
                nn.functional.cross_entropy(logits, y).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

            step()   # creates Adam state
            sync(dev)
            torch.cuda.reset_peak_memory_stats()
            t = median_time(step, dev, reps=10, warmup=2)
            peak = torch.cuda.max_memory_allocated()
            static = sum(p.numel() for p in model.parameters()) * 4 * 4
            R(f"{name:>14} {'on' if ckpt else 'off':>10} {peak / 2**30:>6.2f}Gi {static / 2**30:>17.2f}Gi {t * 1e3:>8.1f}")
            res[f"{name}{' ckpt' if ckpt else ''}"] = {"peak_bytes": peak, "static_bytes": static, "step_s": t}
            del model, opt, x
            torch.cuda.empty_cache()
    out["cuda_training_step"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--skip-memory", action="store_true")
    args = ap.parse_args()
    set_threads()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tag = "cuda" if dev.type == "cuda" else "cpu"
    R = Report(f"07_time_and_space_{tag}.txt", device=args.device)
    out = {}
    latency_table(R, dev, 1, args.reps, out)
    operator_table(R, dev, 8 if dev.type == "cpu" else 64, args.reps, out)
    if dev.type == "cuda":
        latency_table(R, dev, 64, args.reps, out)
        cuda_training_step(R, dev, out)
    elif not args.skip_memory:
        memory_table(R, out)
    (RESULTS / f"07_time_and_space_{tag}.json").write_text(json.dumps(out, indent=1))
    R.close()


if __name__ == "__main__":
    main()
