"""Answer priors of the counting task: what a constant answer scores.

Counting RL is only a counting gain if it beats the best constant answer. This script prints the answer
distribution of the held-out counting questions (the set train_grpo.py evaluates on) and of the training
images the default run sees, so the reader can compare every counting result with "always say 1".

    python c09-vlm-post-training/scripts/majority_baseline.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmalign.evaluate import vqa_items  # noqa: E402
from vlmalign.rl_tasks import count_examples  # noqa: E402


def show(name: str, answers: list[str]) -> None:
    c = Counter(answers)
    n = len(answers)
    best, k = max(c.items(), key=lambda kv: (kv[1], kv[0]))
    dist = ", ".join(f"'{a}': {v}" for a, v in sorted(c.items()))
    print(f"{name}: n={n}  {dist}  | best constant answer '{best}' scores {k / n:.3f}")


def main() -> None:
    held = [q.answer for _, q in vqa_items(100, 12345, kinds=("count",))]
    show("held-out counting questions (100, seed 12345)", held)
    steps, prompts = 60, 4
    train = [e.answer for e in count_examples(steps * prompts, seed=1)]  # train_grpo.py default: seed 0 + 1
    show("training images, all 60 steps (240)", train)
    show("training images, first 8 steps (32)", train[: 8 * prompts])
    show("training images, last 8 steps (32)", train[-8 * prompts:])


if __name__ == "__main__":
    main()
