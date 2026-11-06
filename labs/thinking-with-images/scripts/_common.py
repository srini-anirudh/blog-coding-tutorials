"""Shared argument parsing for the model scripts."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))


def parser(doc: str, n: int) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--n", type=int, default=n, help="number of scenes")
    ap.add_argument("--stub", action="store_true", help="use the stub reader instead of Qwen2-VL (offline check)")
    ap.add_argument("--out", default=None, help="results path without suffix")
    return ap


def reader(args, attn="sdpa"):
    if args.stub:
        from lookagain.stub import StubVLM
        return StubVLM()
    from lookagain.vlm import QwenVLM
    return QwenVLM(attn=attn)


def command(script: str) -> str:
    return "python labs/thinking-with-images/scripts/" + script + "".join(" " + shlex.quote(a) for a in sys.argv[1:])


def out_path(args, name: str) -> Path:
    if args.out:
        return Path(args.out)
    return LAB / "results" / (name + ("_stub" if args.stub else ""))
