"""Memory diagnosis (Figure 07): allocated, reserved, peak, and an OOM while gigabytes sit free.

Part 1 shows the caching allocator keep freed memory (reserved > allocated) and the peak that decides
whether you fit. Part 2 caps the process at 2 GiB, fills it with 4 MiB tensors (the allocator packs five of
them into each 20 MiB segment), frees four of every five, then asks for 1 GiB. With the default allocator
that fails although 1.25 GiB is cached but idle, because every segment still holds one live tensor.
We run the same program with PYTORCH_ALLOC_CONF=expandable_segments:True (on this torch build it still fails),
and once more with the long-lived tensors allocated first. Then the temporaries fill segments of their own,
those segments empty completely, the allocator returns them on the retry, and the 1 GiB request succeeds.

    python scripts/06_memory.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whyslow.bench import Report, need_cuda  # noqa: E402
from whyslow.memory import diagnose_oom, gib, parse_oom, stats  # noqa: E402

CAP = 2 << 30


def child(stable: bool) -> None:
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(CAP / total)
    if stable:  # the 80 long-lived tensors first, then the 320 temporaries
        keep = [torch.empty(4 << 20, dtype=torch.uint8, device="cuda") for _ in range(80)]
        tmp = [torch.empty(4 << 20, dtype=torch.uint8, device="cuda") for _ in range(320)]
        del tmp
    else:  # interleaved: one long-lived tensor in every group of five
        small = [torch.empty(4 << 20, dtype=torch.uint8, device="cuda") for _ in range(400)]
        keep = small[::5]
        del small
    before = stats()
    out = {"before": before}
    try:
        big = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
        out["result"] = "ok"
        out["after"] = stats()
        del big
    except torch.OutOfMemoryError as e:
        f = parse_oom(str(e))
        out |= {"result": "OOM", "oom": f, "diagnosis": diagnose_oom(f, limit=CAP),
                "message": re.sub(r"Process \d+", "Process <pid>", str(e).split(". If")[0])}
    del keep
    print("JSON" + json.dumps(out))


def run_child(conf: str | None, stable: bool = False) -> dict:
    env = dict(os.environ)
    for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        env.pop(k, None)
        if conf:
            env[k] = conf
    p = subprocess.run([sys.executable, __file__, "--child", *(["--stable"] if stable else [])], env=env, capture_output=True, text=True)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("JSON")]
    if not line:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(line[0][4:])


def main():
    r = Report("06_memory")
    if not need_cuda(r):
        return r.save()
    r("Part 1: allocated, reserved, peak")
    torch.cuda.reset_peak_memory_stats()
    keep = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    tmp = torch.empty(2 << 30, dtype=torch.uint8, device="cuda")
    s1 = stats()
    del tmp
    s2 = stats()
    torch.cuda.empty_cache()
    s3 = stats()
    for label, s in [("1 GiB kept + 2 GiB temporary", s1), ("after del temporary", s2), ("after empty_cache()", s3)]:
        r(f"  {label:30s} allocated {gib(s['allocated'])}  reserved {gib(s['reserved'])}  peak {gib(s['peak'])}")
    del keep
    torch.cuda.empty_cache()

    r(f"\nPart 2: a {gib(CAP)} cap, 400 x 4 MiB allocated, 80 kept, then ask for 1 GiB")
    res = {}
    for name, conf, stable in [("interleaved, default", None, False),
                               ("interleaved, expandable_segments", "expandable_segments:True", False),
                               ("long-lived first, default", None, True)]:
        c = run_child(conf, stable)
        res[name] = c
        b = c["before"]
        r(f"  {name:33s} before: allocated {gib(b['allocated'])}, reserved {gib(b['reserved'])}, "
          f"cached idle {gib(b['cached_idle'])}  ->  {c['result']}")
        if c["result"] == "OOM":
            f = c["oom"]
            r(f"    OOM numbers: requested {gib(f['requested'])}, free on device {gib(f.get('free', 0))}, "
              f"allocated {gib(f['allocated'])}, reserved but unallocated {gib(f['reserved_unallocated'])}")
            r(f"    verdict: {c['diagnosis']}")
            r(f"    message: {c['message']}")
        else:
            a = c["after"]
            r(f"    after: allocated {gib(a['allocated'])}, reserved {gib(a['reserved'])}")
    r.data = {"part1": {"with_temp": s1, "after_del": s2, "after_empty_cache": s3}, "part2": res}
    return r.save()


if __name__ == "__main__":
    if "--child" in sys.argv:
        child("--stable" in sys.argv)
    else:
        main()
