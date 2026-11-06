"""Section 3: measure T(n) between two processes and fit T = alpha + beta n.

    python scripts/02_pingpong.py                  # two CPU processes over gloo (TCP loopback)
    python scripts/02_pingpong.py --backend nccl   # two GPUs over NCCL (needs 2 visible GPUs)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commcost import alphabeta as ab  # noqa: E402
from commcost import dist_run, workers  # noqa: E402
from commcost.report import Report, human_bytes, table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--max-log2", type=int, default=None, help="largest message is 2**this bytes")
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()
    top = args.max_log2 or (30 if args.backend == "nccl" else 26)
    sizes = [1 << k for k in range(2, top + 1, 2)]

    rep = Report(f"02_pingpong_{args.backend}", gpu=args.backend == "nccl",
                 note="2 processes" + (", 2 GPUs" if args.backend == "nccl" else ", 1 thread each"))
    rows = dist_run.launch(2, workers.pingpong, (sizes, args.iters, args.backend), backend=args.backend)
    n = [r["bytes"] for r in rows]
    t = [r["one_way_s"] for r in rows]
    alpha, beta = ab.fit_alpha_beta(n, t)
    rep(f"Fit over {len(n)} sizes, weighted by 1/T: alpha = {alpha * 1e6:.2f} us, 1/beta = {1 / beta / 1e9:.2f} GB/s, "
        f"n_half = {human_bytes(ab.n_half(alpha, beta))}")
    rep()
    out = []
    for size, tt in zip(n, t):
        model = ab.message_time(size, alpha, beta)
        out.append([human_bytes(size), f"{tt * 1e6:,.1f}", f"{model * 1e6:,.1f}", f"{(tt / model - 1):+.0%}",
                    f"{size / tt / 1e9:.3f}", f"{size / tt * beta:.0%}"])
    for line in table(out, ["message", "measured us", "model us", "error", "GB/s", "of peak"]):
        rep(line)
    rep()
    rep("Below n_half the time barely moves with size; above it, time grows in proportion to bytes.")
    rep.save({"alpha_s": alpha, "beta_s_per_byte": beta, "rows": rows})


if __name__ == "__main__":
    main()
