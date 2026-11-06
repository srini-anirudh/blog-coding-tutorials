"""A queue that runs dry (Figure 05).

Part 1 simulates the bounded prefetch queue with the post's numbers: a GPU that wants a batch every
80 ms, fed by a pool that delivers one every 96 ms (4 workers at 384 ms each), then by 8 workers.
Part 2 runs a real torch DataLoader whose workers sleep to stand in for decode, and a consumer that sleeps
80 ms to stand in for the GPU step, and times next(iter) separately from the step, as the post suggests.
Sleep keeps the result independent of how loaded the CPU is.

    python scripts/04_input_pipeline.py            # full run, about 40 s
    python scripts/04_input_pipeline.py --quick    # shorter real-loader part
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whyslow.bench import Report  # noqa: E402
from whyslow.queue_sim import simulate, steady_state, steps_until_dry  # noqa: E402


class SlowDecode(Dataset):
    """Each item costs `ms_per_item` of sleep, so a batch of `batch` items costs batch * ms_per_item."""

    def __init__(self, n: int, ms_per_item: float):
        self.n, self.ms = n, ms_per_item

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        time.sleep(self.ms / 1e3)
        return torch.full((16,), float(i))


def real_loader(workers: int, batch_ms: float, gpu_ms: float, steps: int, skip: int, batch: int = 8) -> dict:
    ds = SlowDecode(steps * batch, batch_ms / batch)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, prefetch_factor=2, persistent_workers=False)
    waits, t_all = [], None
    it = iter(dl)
    for j in range(steps):
        t = time.perf_counter()
        next(it)
        waits.append((time.perf_counter() - t) * 1e3)
        if j == skip:
            t_all = time.perf_counter()
        time.sleep(gpu_ms / 1e3)  # the GPU step
    total = (time.perf_counter() - t_all) * 1e3
    w = waits[skip + 1:]
    n = len(w)
    return {"mean_wait_ms": sum(w) / n, "step_ms": total / n, "wait_fraction": sum(w) / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    r = Report("04_input_pipeline" + ("_quick" if args.quick else ""))
    gpu_ms, per_worker_ms = 80.0, 384.0

    r("Part 1: simulation. GPU step 80 ms. prefetch_factor 2, so up to 2 x workers batches in flight.\n")
    r(f"{'config':44s} {'dry after':>9s} {'wait/step':>9s} {'time lost':>9s} {'mean depth':>10s}  depth histogram")
    r.data["sim"] = {}
    for name, w in [("A  4 workers, one batch per 96 ms", 4), ("B  8 workers, one batch per 48 ms", 8)]:
        res = simulate(w, per_worker_ms, gpu_ms, prefetch=2, steps=400, prefill=True)
        s = steady_state(res)
        dry = steps_until_dry(res)
        lost = s["mean_wait_ms"] / (gpu_ms + s["mean_wait_ms"])
        hist = " ".join(f"{k}:{v}" for k, v in s["depth_hist"].items() if v)
        r(f"{name:44s} {str(dry) if dry is not None else 'never':>9s} {s['mean_wait_ms']:8.1f}ms {lost:9.1%} "
          f"{s['mean_depth']:10.2f}  {hist}")
        r.data["sim"][name] = {"dry_after_steps": dry, **s, "time_lost": lost}
    r("Depth counts batches ready when the GPU asks. The queue starts full; A runs dry and stays dry.")

    steps, skip = (40, 12) if args.quick else (120, 30)
    r(f"\nPart 2: real DataLoader, {steps} steps, first {skip} skipped. Workers sleep 384 ms per batch of 8.\n")
    r(f"{'workers':>7s} {'wait in next()':>15s} {'step':>9s} {'share of time waiting':>22s}")
    r.data["real"] = {}
    for w in (4, 8):
        res = real_loader(w, per_worker_ms, gpu_ms, steps, skip)
        r(f"{w:7d} {res['mean_wait_ms']:13.1f}ms {res['step_ms']:7.1f}ms {res['wait_fraction']:22.1%}")
        r.data["real"][w] = res
    return r.save()


if __name__ == "__main__":
    main()
