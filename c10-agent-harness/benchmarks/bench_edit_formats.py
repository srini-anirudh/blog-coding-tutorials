"""Compare edit formats under the slips real models make.

Each trial picks a real line of real Python and asks for a one-line change. We then
write the edit as a model would in each format, with one kind of slip applied, and
apply it. Three outcomes:

    correct    the file now equals the intended result
    refused    the applier raised a clear error (the model can retry)
    corrupted  the applier succeeded and the file is wrong (nobody is told)

The slips are simulated, not measured from a model. They show what each format does
when a slip happens, not how often a slip happens.

    python c10-agent-harness/benchmarks/bench_edit_formats.py --trials 200
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from harness.editing import EditError, apply_hunks, apply_search_replace, make_diff, parse_unified_diff  # noqa: E402
from harness.tokens import count_tokens  # noqa: E402

FORMATS = ["whole file", "search/replace", "strict diff", "anchored diff"]
SLIPS = ["clean", "stale line numbers", "shifted indentation", "flattened indentation", "quote too short",
         "lazy rewrite", "misremembered line"]


def corpus() -> list[str]:
    files = sorted((HERE / "harness").rglob("*.py"))
    return [t for t in (f.read_text() for f in files) if len(t.splitlines()) > 60]


def pick(rng: random.Random, texts: list[str], duplicate: bool) -> tuple[str, int]:
    """Choose a file and a line index. duplicate=True wants a line occurring more than once."""
    while True:
        text = rng.choice(texts)
        lines = text.splitlines()
        i = rng.randrange(5, len(lines) - 5)
        stripped = lines[i].strip()
        if len(stripped) < 8 or stripped.startswith(("#", '"""')):
            continue
        if (lines.count(lines[i]) > 1) == duplicate:
            return text, i


def run_format(fmt: str, view: str, actual: str, i: int, slip: str, rng: random.Random) -> tuple[str, int]:
    """Write the edit from the model's `view`, apply it to `actual`. Returns (output text, cost)."""
    lines = view.splitlines()
    new_line = lines[i] + "  # edited"
    target_view = "\n".join(lines[:i] + [new_line] + lines[i + 1:]) + "\n"

    if fmt == "whole file":
        payload = target_view
        if slip == "lazy rewrite":
            keep = target_view.splitlines()[: max(i + 3, len(lines) // 2)]
            payload = "\n".join(keep) + "\n# ... rest of the file unchanged ...\n"
        if slip == "misremembered line":
            other = rng.choice([k for k in range(len(lines)) if k != i and lines[k].strip()])
            out = payload.splitlines()
            out[other] = out[other].replace("e", "a", 1) if "e" in out[other] else out[other] + " "
            payload = "\n".join(out) + "\n"
        return payload, count_tokens(payload)

    a, b = max(0, i - 3), min(len(lines), i + 4)
    if fmt == "search/replace":
        old = lines[i:i + 1] if slip == "quote too short" else lines[max(0, i - 1):i + 2]
        new = [new_line] if slip == "quote too short" else lines[max(0, i - 1):i] + [new_line] + lines[i + 1:i + 2]
        if slip == "shifted indentation":
            old, new = ["    " + ln for ln in old], ["    " + ln for ln in new]
        if slip == "flattened indentation":
            old, new = [ln.strip() for ln in old], [ln.strip() for ln in new]
        if slip == "misremembered line":
            old = [ln.replace("e", "a", 1) for ln in old]
        search, replace = "\n".join(old), "\n".join(new)
        return apply_search_replace(actual, search, replace).text, count_tokens(search) + count_tokens(replace)

    # Both diff formats start from the same text; `strict` only changes how it is applied.
    diff = make_diff(view, target_view, "f.py")
    if slip in ("shifted indentation", "flattened indentation"):
        out = []
        for ln in diff.splitlines():
            if ln[:1] in " +-" and not ln.startswith(("---", "+++")):
                ln = ln[0] + ("    " + ln[1:] if slip.startswith("shifted") else ln[1:].lstrip())
            out.append(ln)
        diff = "\n".join(out) + "\n"
    if slip == "misremembered line":
        out = diff.splitlines()
        ctx = [k for k, ln in enumerate(out) if ln.startswith(" ") and "e" in ln]
        if ctx:
            k = rng.choice(ctx)
            out[k] = out[k].replace("e", "a", 1)
        diff = "\n".join(out) + "\n"
    hunks = parse_unified_diff(diff)[0].hunks
    return apply_hunks(actual, hunks, strict=fmt == "strict diff").text, count_tokens(diff)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    texts = corpus()

    print(f"{args.trials} trials per cell, one-line edits in {len(texts)} real Python files\n")
    costs: dict[str, list[int]] = {f: [] for f in FORMATS}
    print(f"{'slip':<24}" + "".join(f"{f:>20}" for f in FORMATS))
    print(f"{'':<24}" + "".join(f"{'ok/refused/CORRUPT':>20}" for _ in FORMATS))
    totals = {f: [0, 0, 0] for f in FORMATS}
    for slip in SLIPS:
        row = f"{slip:<24}"
        for fmt in FORMATS:
            rng = random.Random(args.seed)   # the same trials for every format
            ok = refused = corrupted = 0
            for _ in range(args.trials):
                view, i = pick(rng, texts, duplicate=slip == "quote too short")
                actual = view
                if slip == "stale line numbers":   # the file grew above the edit since the model read it
                    actual = "# inserted one\n# inserted two\n# inserted three\n" + view
                lines = actual.splitlines()
                j = i + (3 if slip == "stale line numbers" else 0)
                expected = "\n".join(lines[:j] + [lines[j] + "  # edited"] + lines[j + 1:]) + "\n"
                try:
                    text, cost = run_format(fmt, view, actual, i, slip, rng)
                    costs[fmt].append(cost)
                except EditError:
                    refused += 1
                    continue
                if text == expected:
                    ok += 1
                else:
                    corrupted += 1
            row += f"{f'{ok}/{refused}/{corrupted}':>20}"
            totals[fmt][0] += ok
            totals[fmt][1] += refused
            totals[fmt][2] += corrupted
        print(row)
    print(f"{'total':<24}" + "".join(f"{'/'.join(map(str, totals[f])):>20}" for f in FORMATS))
    print(f"\n{'mean tokens per edit':<24}" + "".join(f"{sum(costs[f]) / max(1, len(costs[f])):>20.0f}" for f in FORMATS))


if __name__ == "__main__":
    main()
