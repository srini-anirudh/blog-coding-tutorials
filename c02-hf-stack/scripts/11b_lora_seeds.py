"""Section 18: how much of the LoRA result is luck. Repeats 11_lora.py across seeds, and repeats one seed exactly.

The held-out sums stay the same in every run. Only the adapter init and the batch order change.

Run: python c02-hf-stack/scripts/11b_lora_seeds.py [--seeds 0 1 2] [--steps 150] [--device auto] [--prefix gpu_]
Writes results/<prefix>11_lora.txt (first seed) and results/<prefix>11_lora_seed<N>.txt (the others). Prints the table.
"""

import argparse
import ast
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--results", type=Path, default=ROOT / "results")
parser.add_argument("--device", default="auto")
parser.add_argument("--prefix", default="", help="prepended to each file name, e.g. gpu_")
args = parser.parse_args()


def run(seed: int) -> tuple[str, dict]:
    command = [sys.executable, str(ROOT / "scripts" / "11_lora.py"), "--seed", str(seed), "--steps", str(args.steps), "--device", args.device]
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    before, after = re.search(r"before ([\d.]+), after ([\d.]+)", out).groups()
    curve = ast.literal_eval(re.search(r"^loss: (.*)$", out, re.M).group(1))
    return out, {
        "before": float(before),
        "after": float(after),
        "curve": curve,
        "held_out": int(re.search(r"on (\d+) held-out", out).group(1)),
        "merge_gap": re.search(r"merged - adapter\| max: (\S+)", out).group(1),
        "sha": re.search(r"adapter sha256: (\w+)", out).group(1),
        "seconds_per_step": re.search(r"\(([\d.]+) s/step\)", out).group(1),
    }


rows = {}
for position, seed in enumerate(args.seeds):
    out, rows[seed] = run(seed)
    name = args.prefix + ("11_lora.txt" if position == 0 else f"11_lora_seed{seed}.txt")
    (args.results / name).write_text(out)

first = args.seeds[0]
_, repeat = run(first)
same = all(repeat[k] == rows[first][k] for k in ("before", "after", "curve", "merge_gap", "sha"))

print(f"{rows[first]['held_out']} held-out sums, {args.steps} steps, exact match within 6 tokens")
print(f"{'seed':>4} {'before':>7} {'after':>7} {'final loss':>11} {'merge gap':>10} {'s/step':>7}  adapter sha256")
final_loss = lambda row: row["curve"][-1][1] if row["curve"] else float("nan")  # noqa: E731
for seed, row in rows.items():
    print(
        f"{seed:>4} {row['before']:>7.3f} {row['after']:>7.3f} {final_loss(row):>11.3f} "
        f"{row['merge_gap']:>10} {row['seconds_per_step']:>7}  {row['sha']}"
    )
scores = [row["after"] for row in rows.values()]
print(f"\nafter: min {min(scores):.3f}, mean {statistics.mean(scores):.3f}, max {max(scores):.3f}")
print(f"seed {first} run a second time, same accuracy, loss curve, merge gap and adapter hash: {same}")
if not same:
    print("second run:", {k: repeat[k] for k in ("after", "merge_gap", "sha")})
