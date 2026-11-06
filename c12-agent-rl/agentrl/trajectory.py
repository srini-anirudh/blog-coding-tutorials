"""Trajectories: tau = (s_0, a_0, r_0, ..., s_T), recorded completely.

A trajectory is stored twice. ``steps`` is the text view: what the agent said and what the tool answered.
``tokens`` is the tensor view: the exact token ids the policy saw and sampled, as one sequence

    <bos> task <act> action_0 <eos> <obs> result_0 <act> action_1 <eos> ...

with ``action_mask`` = 1 on sampled action tokens and 0 on everything the environment wrote. Keeping the sampled
ids (instead of re-tokenising the text later) matters: with a BPE tokenizer, decode-then-encode does not always
give back the same ids, and then the log-probabilities used in the update no longer match the ones sampled.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Step:
    action: str            # what the agent emitted, e.g. "search port"
    observation: str       # what the tool returned, e.g. "found db/main"
    reward: float = 0.0    # outcome reward arrives on the last step only
    forced: bool = False   # True for injected mistakes (perturbed starts); never trained on
    error: bool = False    # the tool returned an error
    progress: float = 0.0  # potential Phi(s_{t+1}) in [0, 1]: fraction of task milestones reached


@dataclass
class Trajectory:
    task_text: str
    level: int
    seed: int
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    submitted: bool = False
    truncated: bool = False        # ran out of steps or context
    # tensor view, filled by the rollout or by ``codec.encode_trajectory``
    tokens: list[int] = field(default_factory=list)
    action_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)   # log pi_old(token) on action tokens, 0 elsewhere
    step_of_token: list[int] = field(default_factory=list)  # step index for action tokens, -1 elsewhere
    state_positions: list[int] = field(default_factory=list)  # index of the <act> token before each step
    group: int = -1                # GRPO group id: trajectories of the same task share it
    policy_version: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def outcome(self) -> float:
        return 1.0 if self.success else 0.0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> "Trajectory":
        d = json.loads(line)
        d["steps"] = [Step(**s) for s in d["steps"]]
        return cls(**d)

    def pretty(self) -> str:
        lines = [f"task [{self.level}]: {self.task_text}"]
        for i, s in enumerate(self.steps):
            tag = " (forced)" if s.forced else ""
            lines.append(f"  {i}: {s.action:<28} -> {s.observation}{tag}")
        lines.append(f"  success={self.success} submitted={self.submitted} truncated={self.truncated}")
        return "\n".join(lines)
