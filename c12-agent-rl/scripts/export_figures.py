"""Real trajectories with per-step advantages, for the replay figure. Writes results/replay.json.

    python c12-agent-rl/scripts/export_figures.py

Groups come from the policy trained with action-level credit, because it has a trained value head. Every number on
the replay figure is computed by the same functions the update uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentrl.advantages import gae, group_relative  # noqa: E402
from agentrl.codec import WordCodec  # noqa: E402
from agentrl.evaluate import Episode  # noqa: E402
from agentrl.grpo import GRPOConfig, state_values  # noqa: E402
from agentrl.online import group_rollouts  # noqa: E402
from agentrl.policy import TinyPolicy  # noqa: E402
from agentrl.rewards import outcome  # noqa: E402


def main() -> None:
    torch.manual_seed(3)
    codec = WordCodec()
    policy = TinyPolicy.load(str(ROOT / "runs" / "grpo_action.pt")).eval()
    cfg = GRPOConfig()
    groups = []
    G = 8
    for level, seed0 in [(4, 500), (5, 700)]:
        episodes = [Episode(level, seed0 + i, []) for i in range(40)]
        trajs = group_rollouts(policy, codec, episodes, G, cfg, "cpu")
        values = state_values(policy, trajs, codec.pad, "cpu")
        R = torch.tensor([t.outcome for t in trajs])
        A = group_relative(R, torch.tensor([t.group for t in trajs]))
        for g in range(len(episodes)):
            idx = [i for i, t in enumerate(trajs) if t.group == g]
            wins = [i for i in idx if trajs[i].success]
            if not 0 < len(wins) < G:
                continue
            members = []
            for i in idx:
                t = trajs[i]
                members.append({
                    "success": t.success,
                    "traj_adv": round(float(A[i]), 3),
                    "steps": [{"action": s.action, "obs": s.observation, "error": s.error,
                               "value": round(values[i][k], 3),
                               "gae": round(gae(outcome(t), values[i], 1.0, 0.95)[k], 3),
                               "td": round(gae(outcome(t), values[i], 1.0, 0.0)[k], 3)}
                              for k, s in enumerate(t.steps)]})
            groups.append({"level": level, "task": trajs[idx[0]].task_text, "wins": len(wins), "members": members})
            if sum(g["level"] == level for g in groups) >= 2:
                break
    (ROOT / "results" / "replay.json").write_text(json.dumps(groups, indent=0))
    print(f"{len(groups)} groups written")
    for g in groups:
        print(f"level {g['level']}: {g['task']} ({g['wins']}/{G} wins)")


if __name__ == "__main__":
    main()
