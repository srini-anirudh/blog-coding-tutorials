"""Print one conversation token by token with its label, and export it for the interactive loss-mask figure.

    python scripts/inspect_mask.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.chat import encode_chat, load_tokenizer, mask_table  # noqa: E402
from posttrain.tasks import add_follow_up, make_problem, to_messages  # noqa: E402
from posttrain.utils import save_json  # noqa: E402


def main() -> None:
    tok = load_tokenizer()
    problem = add_follow_up(random.Random(1), make_problem(random.Random(0), "chain"))
    messages = to_messages(problem)
    example = encode_chat(tok, messages)
    rows = mask_table(tok, example)

    print(f"{'pos':>3}  {'id':>6}  {'token':<16} {'role':<10} label")
    for r in rows:
        print(f"{r['pos']:>3}  {r['id']:>6}  {r['text']!r:<16} {r['role']:<10} {r['label']}")
    targets = sum(r["label"] != -100 for r in rows)
    print(f"\n{len(rows)} tokens, {targets} targets, {len(rows) - targets} masked")

    variants = {m: encode_chat(tok, messages, mode=m).labels for m in ("assistant", "last", "all")}
    save_json({"tokens": rows, "labels": variants}, "mask_example.json")


if __name__ == "__main__":
    main()
