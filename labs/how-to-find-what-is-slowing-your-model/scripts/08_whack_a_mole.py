"""Case study: whack-a-mole (Figure 09), on a small run that is slow for three reasons at once.

A residual MLP (width 1024, depth 8, batch 4096) trains on batches from a DataLoader whose workers spend
120 ms building each batch. The baseline has 2 workers, pageable memory, fp32 matmuls, a per-parameter AdamW
and a loss.item() every step. At each stage we measure step time, time waiting in next(), the GPU busy
time and kernels per step from a profiler trace. A fixed rule turns those numbers into a diagnosis, and
the next stage applies the fix the diagnosis names:

    0 baseline
    1 input pipeline: 32 workers, pinned memory, non_blocking copy
    2 compute: bf16 autocast, so the GEMMs run on tensor cores
    3 launch overhead: torch.compile the model, fused AdamW, read the loss every 50 steps
    4 launch overhead again, if the diagnosis still says so: CUDA graphs (torch.compile mode="reduce-overhead")
and stop when the diagnosis says the step is dense GPU work, because what is left needs a different model,
numerics or parallelism.

    python scripts/08_whack_a_mole.py
"""

import itertools
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whyslow.bench import Report, need_cuda  # noqa: E402
from whyslow.profile import profile_steps  # noqa: E402
from whyslow.roofline import H100_SXM  # noqa: E402
from whyslow.timeline import max_speedup  # noqa: E402
from whyslow.toy import ResMLP, flops_per_step  # noqa: E402

D, DEPTH, BATCH, BATCH_MS = 1024, 8, 4096, 120.0

STAGES = [
    ("0 baseline", dict(workers=2, pin=False, compiled=False, fused=False, bf16=False, log_every=1)),
    ("1 + input pipeline", dict(workers=32, pin=True, compiled=False, fused=False, bf16=False, log_every=1)),
    ("2 + bf16", dict(workers=32, pin=True, compiled=False, fused=False, bf16=True, log_every=1)),
    ("3 + launch overhead", dict(workers=32, pin=True, compiled=True, fused=True, bf16=True, log_every=50)),
    ("4 + CUDA graphs", dict(workers=32, pin=True, compiled="reduce-overhead", fused=True, bf16=True, log_every=50)),
]


class Batches(Dataset):
    """Whole batches, as int8 to keep the host-to-device copy small. Building one costs BATCH_MS: it stands in
    for decode and augmentation."""

    def __len__(self):
        return 1 << 20

    def __getitem__(self, i):
        t = time.perf_counter()
        x = torch.randint(-128, 128, (BATCH, D), dtype=torch.int8)
        y = torch.randint(0, 16, (BATCH,))
        time.sleep(max(0.0, BATCH_MS / 1e3 - (time.perf_counter() - t)))
        return x, y


def build(cfg):
    torch.manual_seed(0)
    model = ResMLP(D, DEPTH).cuda()
    c = cfg["compiled"]
    fwd = torch.compile(model, mode=c) if isinstance(c, str) else torch.compile(model) if c else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, **({"fused": True} if cfg["fused"] else {"foreach": False}))
    loss_fn = nn.CrossEntropyLoss()
    dl = DataLoader(Batches(), batch_size=None, num_workers=cfg["workers"], pin_memory=cfg["pin"],
                    prefetch_factor=2, persistent_workers=True)
    it = iter(dl)
    counter = itertools.count()
    wait = []

    def step():
        t = time.perf_counter()
        x, y = next(it)
        wait.append(time.perf_counter() - t)
        x, y = x.cuda(non_blocking=cfg["pin"]).float() / 64, y.cuda(non_blocking=cfg["pin"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["bf16"]):
            loss = loss_fn(fwd(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if next(counter) % cfg["log_every"] == 0:
            loss.item()

    return step, wait, dl


def measure(cfg, steps=60, warmup=20):
    step, wait, dl = build(cfg)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    wait.clear()
    t = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t) / steps * 1e3
    data_ms = sum(wait) / len(wait) * 1e3
    s = profile_steps(step, active=10, warmup=2, wait=1)
    del dl
    # The profiler slows a launch-bound step (the post's first Trap), so the busy fraction that drives the
    # diagnosis divides kernel time from the trace by the unprofiled step time. Both are reported.
    gpu_ms = s["gpu_busy_us_per_step"] / 1e3
    return {"step_ms": step_ms, "data_wait_ms": data_ms, "busy_fraction": min(1.0, gpu_ms / step_ms),
            "busy_fraction_in_trace": s["busy_fraction"], "profiled_step_ms": s["step_us"] / 1e3,
            "kernels_per_step": s["kernels"] / len(s["steps"]), "mean_kernel_us": s["mean_us"],
            "gpu_ms": s["gpu_busy_us_per_step"] / 1e3, "top_ops": s["top_ops"]}


def diagnose(m: dict, cfg: dict) -> str:
    """The diagnosis tree, cut down to the rows this run can be in."""
    if m["data_wait_ms"] > 0.2 * m["step_ms"]:
        return "input pipeline: the GPU waits in next()"
    if m["busy_fraction"] < 0.7:
        return "gaps: CPU launch overhead"
    if not cfg["bf16"]:
        return "dense GPU work in fp32: tensor cores idle"
    return "dense GPU work: stop"


def main():
    r = Report("08_whack_a_mole")
    if not need_cuda(r):
        return r.save()
    flops = flops_per_step(D, DEPTH, BATCH)
    r(f"ResMLP d={D} depth={DEPTH} batch={BATCH}; {flops / 1e9:.0f} GFLOP per step (6 x weights x batch). "
      f"MFU against {H100_SXM['bf16_flops'] / 1e12:.0f} TFLOP/s dense bf16, the post's harsh denominator.\n")
    r(f"{'stage':22s} {'step ms':>8s} {'profiled':>8s} {'data wait':>9s} {'GPU busy':>8s} {'busy frac':>9s} "
      f"{'kernels':>7s} {'mean kern':>9s} {'MFU':>6s} {'speedup':>7s}  diagnosis")
    rows, base, last = {}, None, None
    for name, cfg in STAGES:
        try:
            m = measure(cfg)
        except Exception as e:  # report a stage that fails instead of losing the others
            r(f"{name:22s} failed: {type(e).__name__}: {str(e)[:200]}")
            continue
        base = base or m["step_ms"]
        m["mfu"] = flops / (m["step_ms"] / 1e3) / H100_SXM["bf16_flops"]
        rows[name] = {k: v for k, v in m.items() if k != "top_ops"} | {"diagnosis": diagnose(m, cfg)}
        r(f"{name:22s} {m['step_ms']:8.2f} {m['profiled_step_ms']:8.2f} {m['data_wait_ms']:7.2f}ms {m['gpu_ms']:6.2f}ms "
          f"{m['busy_fraction']:9.2f} {m['kernels_per_step']:7.0f} {m['mean_kernel_us']:7.1f}us {m['mfu']:6.1%} "
          f"{base / m['step_ms']:6.2f}x  {diagnose(m, cfg)}")
        r.data[name + " top ops"] = m["top_ops"]
        last = name
    r("\nstep ms: unprofiled wall clock. profiled: the same step under torch.profiler. busy frac: kernel time from")
    r("the trace divided by the unprofiled step, because the profiler slows a launch-bound step.")
    lb = rows[last]["busy_fraction"]
    r(f"After the last stage the GPU is busy {lb:.0%} of the step: removing every gap could buy at most "
      f"{max_speedup(1 - lb) if lb < 1 else 1.0:.2f}x.")
    r("Top operators by self CUDA time, last stage:")
    r(r.data[last + " top ops"])
    r.data["stages"] = rows
    return r.save()


if __name__ == "__main__":
    main()
