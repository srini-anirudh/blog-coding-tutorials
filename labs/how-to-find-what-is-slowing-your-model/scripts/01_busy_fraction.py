"""Triage step 1: sum of kernel time divided by wall time.

Two training steps of the same residual MLP. One is small (width 256, batch 64): thousands of short kernels
and a GPU lane full of holes. One is large (width 4096, batch 8192, bf16): dense GEMMs back to back.
For each we print the GPU busy fraction, the largest gap, kernel count and mean duration, the Amdahl
ceiling 1 / busy fraction, the top operators by self CUDA time, and the profiler's own overhead.

    python scripts/01_busy_fraction.py
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whyslow.bench import Report, need_cuda  # noqa: E402
from whyslow.profile import profile_steps, unprofiled_step_ms  # noqa: E402
from whyslow.toy import ResMLP, flops_per_step, make_step  # noqa: E402

CASES = {
    "small (d=256, depth=12, batch=64, fp32)": dict(d=256, depth=12, batch=64, dtype=None),
    "large (d=4096, depth=12, batch=8192, bf16)": dict(d=4096, depth=12, batch=8192, dtype=torch.bfloat16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=None, help="folder to keep the chrome traces (open in Perfetto)")
    args = ap.parse_args()
    r = Report("01_busy_fraction")
    if not need_cuda(r):
        return r.save()
    torch.manual_seed(0)
    r.data["cases"] = {}
    for name, c in CASES.items():
        model = ResMLP(c["d"], c["depth"]).cuda()
        x = torch.randn(c["batch"], c["d"], device="cuda")
        y = torch.randint(0, 16, (c["batch"],), device="cuda")
        step = make_step(model, x, y, dtype=c["dtype"])
        plain = unprofiled_step_ms(step)
        trace = Path(args.traces) / f"{name.split()[0]}.json" if args.traces else None
        s = profile_steps(step, trace_path=trace)
        prof_ms = s["step_us"] / 1e3
        tflops = flops_per_step(c["d"], c["depth"], c["batch"]) / (plain / 1e3) / 1e12
        r(f"\n== {name}")
        r(f"step time, unprofiled          {plain:8.2f} ms")
        r(f"step time, under the profiler  {prof_ms:8.2f} ms  ({100 * (prof_ms / plain - 1):+.0f}%)")
        r(f"GPU busy per step              {s['gpu_busy_us_per_step'] / 1e3:8.2f} ms")
        r(f"GPU busy fraction in the trace {s['busy_fraction']:8.3f}  -> kernel work can buy at most "
          f"{1 / s['busy_fraction']:.2f}x")
        r(f"GPU busy / unprofiled step     {s['gpu_busy_us_per_step'] / 1e3 / plain:8.3f}  (the profiler slows the CPU lane, "
          f"not the kernels)")
        r(f"kernels per step               {s['kernels'] / len(s['steps']):8.0f}")
        r(f"mean / median kernel           {s['mean_us']:8.1f} / {s['median_us']:.1f} us")
        r(f"largest GPU gap in a step      {max(p['largest_gap_us'] for p in s['steps']) / 1e3:8.3f} ms")
        r(f"model TFLOP/s (6 x weights)    {tflops:8.2f}")
        r("top operators by self CUDA time:")
        r(s["top_ops"])
        r.data["cases"][name] = {k: v for k, v in s.items() if k not in ("top_ops",)} | {
            "unprofiled_ms": plain, "profiled_ms": prof_ms, "tflops": tflops}
        del model, x, y, step
        torch.cuda.empty_cache()
    return r.save()


if __name__ == "__main__":
    main()
