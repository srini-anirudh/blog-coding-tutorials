"""Print what the registry generates, and what each tool costs in context.

    python c10-agent-harness/scripts/show_schemas.py [--tokenizer Qwen/Qwen2.5-0.5B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import coding_tools  # noqa: E402
from harness.fs_tools import read_file  # noqa: E402
from harness.messages import dumps  # noqa: E402
from harness.tokens import count_tokens  # noqa: E402
from harness.toolset import PRESETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()
    print(json.dumps(read_file.__tool__.schema(), indent=2))

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def real(text: str) -> str:
        return f"{len(tok(text).input_ids):>6}" if tok else "     -"

    print(f"\n{'preset':<14}{'tools':>6}{'estimate':>10}{'tokenizer':>11}")
    for name in PRESETS:
        reg = coding_tools(name)
        blob = dumps(reg.schemas())
        print(f"{name:<14}{len(reg.names):>6}{count_tokens(blob):>10}{real(blob):>11}")

    print(f"\n{'tool':<24}{'estimate':>10}{'tokenizer':>11}")
    reg = coding_tools()
    for s in reg.schemas():
        blob = dumps(s)
        print(f'{s["function"]["name"]:<24}{count_tokens(blob):>10}{real(blob):>11}')

    if tok:
        samples = {"english prose": ("The agent reads the file, finds the bug, and runs the tests again. " * 20),
                   "python code": (Path(__file__).resolve().parents[1] / "harness" / "agent.py").read_text(),
                   "json schemas": dumps(reg.schemas())}
        print(f"\n{'text':<16}{'estimate':>10}{'tokenizer':>11}{'ratio':>8}")
        for name, text in samples.items():
            est, rl = count_tokens(text), len(tok(text).input_ids)
            print(f"{name:<16}{est:>10}{rl:>11}{est / rl:>8.2f}")


if __name__ == "__main__":
    main()
