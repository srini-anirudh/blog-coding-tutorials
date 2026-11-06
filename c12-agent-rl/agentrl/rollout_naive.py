"""Reference rollout: one episode, one token at a time, batch size 1. Easy to read, slow to run."""

from __future__ import annotations

import torch

from .codec import OBS_BUDGET, Codec, observation_ids, prompt_ids
from .env import FileSystemEnv
from .sampling import append, finish, new_trajectory, sample_token
from .trajectory import Trajectory


@torch.no_grad()
def rollout_one(policy, codec: Codec, env: FileSystemEnv, *, temperature: float = 1.0,
                forced: list[str] | None = None, max_action_tokens: int = 6, max_len: int = 384,
                device: str = "cpu") -> Trajectory:
    tr = new_trajectory(env, prompt_ids(codec, env.reset()))
    forced = list(forced or [])
    while not env.done():
        step = len(env.history)
        tr.state_positions.append(len(tr.tokens) - 1)   # the <act> token: the state the action is taken in
        is_forced = bool(forced)
        if is_forced:                                      # an injected mistake: played, never trained on
            text = forced.pop(0)
            append(tr, codec.encode(text) + [codec.eos], None, step)
        else:
            ids, lps = [], []
            while len(ids) < max_action_tokens:
                x = torch.tensor([tr.tokens + ids], device=device)          # (1, T)
                tok, lp = sample_token(policy(x).logits[:, -1], temperature)  # next-token logits (1, V)
                ids.append(int(tok))
                lps.append(float(lp))
                if ids[-1] == codec.eos:
                    break
            append(tr, ids, lps, step)
            text = codec.decode([i for i in ids if i != codec.eos])
        obs, _, done = env.step(text)
        env.history[-1].forced = is_forced
        if done:
            break
        if len(tr.tokens) + max_action_tokens + OBS_BUDGET > max_len:  # out of context: end the episode unsubmitted
            break
        append(tr, observation_ids(codec, obs), None, step)
    return finish(tr, env)
