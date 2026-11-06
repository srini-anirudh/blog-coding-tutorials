"""What PPO costs you: persistent training state for PPO-based RLHF and for GRPO with a verifier.

    python labs/how-reinforcement-learning-teaches-models-to-reason/scripts/04_ppo_memory.py
"""

import _path  # noqa: F401
import torch

from rlvrlab.common import Report
from rlvrlab.memory import GRPO_RLVR, PPO_RLHF, measure_bytes_per_param, persistent_state_gb


def main() -> None:
    rep = Report("04_ppo_memory")
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 512))
    fp32 = measure_bytes_per_param(model, mixed_precision=False)
    model = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 512))
    mixed = measure_bytes_per_param(model, mixed_precision=True)
    rep.print(f"Measured bytes per trained parameter after one AdamW step: fp32 {fp32:.2f}, bf16 + fp32 master {mixed:.2f}")
    rep.print()
    rows = []
    for name, models in (("PPO-based RLHF", PPO_RLHF), ("GRPO with a verifier", GRPO_RLVR)):
        gb = persistent_state_gb(models, 7e9)
        for part, v in gb.items():
            rows.append([name, part, round(v, 1)])
    rep.print("7B parameters per model, 16 B per trained parameter, 2 B per frozen one:")
    rep.table(["pipeline", "model", "GB"], rows)
    ppo = persistent_state_gb(PPO_RLHF, 7e9)
    grpo = persistent_state_gb(GRPO_RLVR, 7e9)
    rep.print()
    rep.print(f"Critic share of PPO state: {ppo['critic'] / ppo['total']:.1%}. GRPO / PPO: {grpo['total'] / ppo['total']:.1%}")
    rep.print("This is arithmetic on stated byte counts, not a measured 7B run. Activations and KV cache are excluded.")
    rep.data = {"bytes_fp32": fp32, "bytes_mixed": mixed, "ppo_gb": ppo, "grpo_gb": grpo}
    rep.save()


if __name__ == "__main__":
    main()
