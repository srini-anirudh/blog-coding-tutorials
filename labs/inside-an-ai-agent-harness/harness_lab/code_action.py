"""One tool call per item, against one action that is a program (post, section 7).

The task: rename `parse_cfg` to `load_cfg` in every Python file of a repository.
Both agents are scripted, so the model's choices are fixed and only the action space
differs. Both really run: the files on disk change, and the two final repositories
are compared byte for byte.

    JSON tools   list_files, then read_file per file, then write_file per match
    code action  one Python program that loops over the files itself

Every model call is charged the whole context, as in the agent loop. Tokens use C10's
estimate (`harness.tokens.count_tokens`).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from harness.tokens import count_tokens

SYSTEM = ("You are a coding agent. Use the tools to complete the task. " * 20).strip()

RENAME_PROGRAM = '''from pathlib import Path
changed = []
for f in sorted(Path(".").glob("**/*.py")):
    s = f.read_text()
    if "parse_cfg" in s:
        f.write_text(s.replace("parse_cfg", "load_cfg"))
        changed.append(str(f))
print(f"renamed in {len(changed)} files: {changed}")
'''


def make_repo(root: Path, n_files: int, seed: int = 0) -> int:
    """Write n small modules. About half call parse_cfg. Returns how many do."""
    rng = random.Random(seed)
    hits = 0
    for i in range(n_files):
        uses = rng.random() < 0.5
        hits += uses
        body = [f'"""Module {i}."""', "", "import os", ""]
        for j in range(6):
            body += [f"def helper_{i}_{j}(x):", f"    return x * {j} + {i}", ""]
        if uses:
            body += ["def setup(path):", "    cfg = parse_cfg(path)", "    return cfg", ""]
        sub = root / f"pkg{i % 4}"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"mod_{i}.py").write_text("\n".join(body) + "\n")
    return hits


@dataclass
class Trace:
    context: list[str] = field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0      # sum over model calls of the context sent
    output_tokens: int = 0

    def model(self, output: str) -> None:
        """One model call: it reads the whole context and writes `output`."""
        self.model_calls += 1
        self.input_tokens += sum(count_tokens(c) for c in self.context)
        self.output_tokens += count_tokens(output)
        self.context.append(output)

    def observe(self, result: str) -> None:
        self.context.append(result)


def call(name: str, **args) -> str:
    return json.dumps({"tool": name, "arguments": args})


def json_tool_agent(root: Path, task: str) -> Trace:
    t = Trace([SYSTEM, task])
    t.model(call("list_files", pattern="**/*.py"))
    files = sorted(str(p.relative_to(root)) for p in root.glob("**/*.py"))
    t.observe(json.dumps(files))
    for f in files:
        t.model(call("read_file", path=f))
        text = (root / f).read_text()
        t.observe(text)
        if "parse_cfg" in text:
            new = text.replace("parse_cfg", "load_cfg")
            t.model(call("write_file", path=f, content=new))
            (root / f).write_text(new)
            t.observe("ok")
    t.model("Renamed parse_cfg to load_cfg everywhere.")
    return t


def code_action_agent(root: Path, task: str) -> Trace:
    t = Trace([SYSTEM, task])
    t.model(call("run_python", code=RENAME_PROGRAM))
    res = subprocess.run([sys.executable, "-I", "-c", RENAME_PROGRAM], cwd=root,
                         capture_output=True, text=True, timeout=60, check=True)
    t.observe(res.stdout.strip())
    t.model("Renamed parse_cfg to load_cfg everywhere.")
    return t


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.glob("**/*.py"))}
