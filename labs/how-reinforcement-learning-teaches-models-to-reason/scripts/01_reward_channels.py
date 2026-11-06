"""Two kinds of reward: the same four rollouts scored by a rule verifier and by a surface-feature proxy.

    python labs/how-reinforcement-learning-teaches-models-to-reason/scripts/01_reward_channels.py
"""

import time

import _path  # noqa: F401

from rlvrlab.common import Report
from rlvrlab.verifier import format_reward, math_reward, surface_proxy

QUESTION = "A box holds 6 rows of 7 eggs. How many eggs?"
TRUTH = "42"
ROLLOUTS = {
    "terse, right": "<think>6*7=42</think> \\boxed{42}",
    "careful, right": "<think>Each row has 7 eggs.\nThere are 6 rows.\n6 x 7 = 42.</think>\nThe answer is \\boxed{42}.",
    "fluent, wrong": ("<think>First, clearly we count the rows.\nThen we multiply each row by the eggs in it.\n"
                      "Therefore 6 x 7 gives the total.\nFinally we check it.</think>\n"
                      "So the answer is clearly \\boxed{48}, therefore 48 eggs."),
    "no format, right": "6 times 7 is 42, so the answer is 42",
}
EXPLOITS = {
    "two boxed answers": "<think>maybe 41</think> \\boxed{41} or \\boxed{42}",
    "answer copied from a leaky prompt": "<think>the prompt says 42</think> \\boxed{42}",
    "empty think block": "<think> </think> \\boxed{42}",
}


def main() -> None:
    rep = Report("01_reward_channels")
    rep.print(f"Question: {QUESTION}  Truth: {TRUTH}")
    rows = [[name, math_reward(t, TRUTH), format_reward(t), surface_proxy(t)] for name, t in ROLLOUTS.items()]
    rep.table(["rollout", "math_reward", "format_reward", "surface_proxy"], rows)
    rep.print()
    rep.print("Ways a rule verifier can still be fooled or can mislead:")
    erows = [[name, math_reward(t, TRUTH), format_reward(t)] for name, t in EXPLOITS.items()]
    rep.table(["case", "math_reward", "format_reward"], erows)

    text = ROLLOUTS["careful, right"]
    n = 20_000
    t0 = time.perf_counter()
    for _ in range(n):
        math_reward(text, TRUTH)
    us = (time.perf_counter() - t0) / n * 1e6
    rep.print()
    rep.print(f"math_reward cost: {us:.2f} microseconds per call ({n} calls, one CPU thread)")
    rep.data = {"rollouts": rows, "exploits": erows, "verifier_us_per_call": us}
    rep.save()


if __name__ == "__main__":
    main()
