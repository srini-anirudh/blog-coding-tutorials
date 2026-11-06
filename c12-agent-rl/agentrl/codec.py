"""Turning the text interface into token ids, and back.

``WordCodec`` is a closed word-level vocabulary built from the environment's pools: every path, key and value is a
single token, so the tiny policy can copy them with one attention lookup. ``HFCodec`` (in ``hf_policy.py``) does
the same job with a pretrained tokenizer. The rollout and the learner only see the ``Codec`` interface.
"""

from __future__ import annotations

from typing import Protocol

from .env import TOOLS
from .tasks import DIRS, KEYS, PATHS, VALUES
from .trajectory import Trajectory

SPECIALS = ["<pad>", "<bos>", "<act>", "<eos>", "<obs>", "<unk>"]
WORDS = (list(TOOLS) + ["set", "in", "to", "value", "of", "and", "found", "none", "ok", "submitted",
                        "error", "internal", "invalid", "input", "no", "such", "file", "key", "dir",
                        "=", ";"])  # "timeout" is also a key, so it comes in with KEYS


class Codec(Protocol):
    pad: int
    bos: int
    act: int     # "your turn": the policy generates after this token
    eos: int     # end of one action
    obs: int     # the environment's answer follows

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class WordCodec:
    def __init__(self) -> None:
        self.vocab = SPECIALS + WORDS + DIRS + PATHS + KEYS + VALUES
        self.index = {w: i for i, w in enumerate(self.vocab)}
        assert len(self.index) == len(self.vocab), "duplicate token"
        self.pad, self.bos, self.act, self.eos, self.obs, self.unk = range(6)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.index.get(w, self.unk) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.vocab[i] for i in ids)


OBS_BUDGET = 16  # longest observation in tokens (a read of a 3-key file is 13); used to stop before the context ends


def prompt_ids(codec: Codec, task_text: str) -> list[int]:
    return [codec.bos] + codec.encode(task_text) + [codec.act]


def observation_ids(codec: Codec, observation: str) -> list[int]:
    return [codec.obs] + codec.encode(observation) + [codec.act]


def encode_trajectory(traj: Trajectory, codec: Codec) -> Trajectory:
    """Fill the tensor view of a text trajectory (used for expert demos and for offline data).

    Layout: <bos> task <act> a_0 <eos> <obs> o_0 <act> a_1 <eos> <obs> o_1 ... ; mask = 1 on a_t and its <eos>.
    """
    tokens = prompt_ids(codec, traj.task_text)
    mask, step_of, states = [0] * len(tokens), [-1] * len(tokens), []
    for t, step in enumerate(traj.steps):
        states.append(len(tokens) - 1)
        action = codec.encode(step.action) + [codec.eos]
        tokens += action
        mask += [0 if step.forced else 1] * len(action)
        step_of += [-1 if step.forced else t] * len(action)
        if t < len(traj.steps) - 1:
            obs = observation_ids(codec, step.observation)
            tokens += obs
            mask += [0] * len(obs)
            step_of += [-1] * len(obs)
    traj.tokens, traj.action_mask, traj.step_of_token, traj.state_positions = tokens, mask, step_of, states
    traj.logprobs = [0.0] * len(tokens)
    return traj
