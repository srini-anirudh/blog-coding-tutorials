"""Play one expert episode per level and print the full trajectory, then show its tensor view.

    python c12-agent-rl/scripts/demo_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrl.batching import collate  # noqa: E402
from agentrl.codec import WordCodec, encode_trajectory  # noqa: E402
from agentrl.env import FileSystemEnv, run_episode  # noqa: E402
from agentrl.expert import ScriptedExpert  # noqa: E402
from agentrl.tasks import sample_task  # noqa: E402


def main() -> None:
    codec = WordCodec()
    for level, seed in [(1, 1), (2, 1), (3, 1), (4, 2), (5, 3)]:
        task = sample_task(level, seed)
        print(run_episode(FileSystemEnv(task), ScriptedExpert(task.text)).pretty(), "\n")

    task = sample_task(2, 1)
    print("files:", task.files)
    tr = encode_trajectory(run_episode(FileSystemEnv(task), ScriptedExpert(task.text)), codec)
    print("\ntoken  id  mask step")
    for t, m, s in zip(tr.tokens, tr.action_mask, tr.step_of_token):
        print(f"{codec.vocab[t]:<10} {t:>3}  {m}  {s:>2}")
    b = collate([tr], codec.pad)
    print("\ntokens", b.tokens.shape, " action_mask", b.action_mask.shape, " actions tokens", int(b.action_mask.sum()))
    print("state positions (the <act> tokens):", tr.state_positions)


if __name__ == "__main__":
    main()
