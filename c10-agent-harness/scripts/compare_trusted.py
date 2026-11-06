"""Check two pieces of the harness against implementations we did not write.

1. The diff applier against `git apply`: same diff, same file, same result?
2. The schema generator against transformers.utils.get_json_schema.

    python c10-agent-harness/scripts/compare_trusted.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from harness.editing import EditError, apply_hunks, make_diff, parse_unified_diff  # noqa: E402
from harness.tools import build_tool  # noqa: E402


def git_apply(text: str, diff: str) -> str | None:
    """Apply `diff` to `text` with real git. Returns the new text, or None if git refuses."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "f.py").write_text(text)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        res = subprocess.run(["git", "apply", "-"], cwd=root, input=diff, text=True, capture_output=True)
        return (root / "f.py").read_text() if res.returncode == 0 else None


def ours(text: str, diff: str, strict: bool) -> str | None:
    try:
        return apply_hunks(text, parse_unified_diff(diff)[0].hunks, strict=strict).text
    except EditError:
        return None


def compare_applier(trials: int = 200, seed: int = 0) -> None:
    files = [t for t in (f.read_text() for f in sorted((HERE / "harness").rglob("*.py"))) if len(t.splitlines()) > 60]
    rng = random.Random(seed)
    cases = {"clean": 0, "stale line numbers": 3}
    print(f"{trials} one-line edits in {len(files)} files, each diff applied by git and by our applier\n")
    print(f"{'case':<20}{'git ok':>8}{'strict ok':>11}{'anchored ok':>13}{'anchored == git':>17}")
    for case, shift in cases.items():
        git_ok = strict_ok = anchored_ok = same = 0
        for _ in range(trials):
            view = rng.choice(files)
            lines = view.splitlines()
            i = rng.randrange(5, len(lines) - 5)
            target = "\n".join(lines[:i] + [lines[i] + "  # edited"] + lines[i + 1:]) + "\n"
            diff = make_diff(view, target, "f.py")
            actual = "# inserted\n" * shift + view
            g, s, a = git_apply(actual, diff), ours(actual, diff, True), ours(actual, diff, False)
            git_ok += g is not None
            strict_ok += s is not None
            anchored_ok += a is not None
            same += a == g
        print(f"{case:<20}{git_ok:>8}{strict_ok:>11}{anchored_ok:>13}{same:>17}")


def sample(path: str, n: int = 3, tags: list[str] | None = None, ratio: float = 0.5) -> str:
    """Do a sample thing.

    Args:
        path: Where to look.
        n: How many lines to show.
        tags: Labels to filter by.
        ratio: A fraction.
    """
    return path


def compare_schema() -> None:
    from transformers.utils import get_json_schema

    mine = build_tool(sample).schema()
    theirs = get_json_schema(sample)
    print("\nschema from get_json_schema:")
    print(json.dumps(theirs, indent=2, sort_keys=True))
    print("\nkeys that differ from ours:")
    a, b = mine["function"]["parameters"], theirs["function"]["parameters"]
    print("  description equal:", mine["function"]["description"] == theirs["function"]["description"])
    print("  required equal:   ", a["required"] == b["required"])
    for name in a["properties"]:
        pa, pb = a["properties"][name], b["properties"][name]
        print(f"  {name:<6} type {pa.get('type')!r} vs {pb.get('type')!r}; only ours: {sorted(set(pa) - set(pb))}; only theirs: {sorted(set(pb) - set(pa))}")
    print("  only ours at parameter level:", sorted(set(a) - set(b)))


if __name__ == "__main__":
    compare_applier()
    compare_schema()
