"""Batched rollout: every live episode generates its next action in the same forward pass.

The reference version (``rollout_naive.py``) runs one forward per token per episode. Here one forward serves all
episodes that are waiting for an action. Sequences are right-padded; with causal attention the padding after a row
cannot leak into it, so we read each row's next-token logits at its own last position.
"""

from __future__ import annotations

import torch

from .codec import OBS_BUDGET, Codec, observation_ids, prompt_ids
from .env import FileSystemEnv
from .sampling import append, finish, new_trajectory, sample_token
from .trajectory import Trajectory


@torch.no_grad()
def sample_actions(policy, prefixes: list[list[int]], codec: Codec, temperature: float,
                   max_tokens: int, device: str) -> list[tuple[list[int], list[float]]]:
    """Generate one action for each prefix. Returns (token ids, log-probs) per row; ids end in <eos> if it fired."""
    B = len(prefixes)
    lens = torch.tensor([len(p) for p in prefixes], device=device)                         # (B,)
    buf = torch.full((B, int(lens.max()) + max_tokens), codec.pad, dtype=torch.long, device=device)
    for i, p in enumerate(prefixes):
        buf[i, : len(p)] = torch.tensor(p, device=device)
    out_tok = torch.full((B, max_tokens), -1, dtype=torch.long, device=device)              # -1 = not generated
    out_lp = torch.zeros(B, max_tokens, device=device)
    alive = torch.arange(B, device=device)                                                  # rows still generating
    for k in range(max_tokens):
        L = int(lens[alive].max())
        logits = policy(buf[alive, :L]).logits                                              # (b, L, V)
        tok, lp = sample_token(logits[torch.arange(len(alive), device=device), lens[alive] - 1], temperature)
        buf[alive, lens[alive]] = tok
        lens[alive] += 1
        out_tok[alive, k], out_lp[alive, k] = tok, lp
        alive = alive[tok != codec.eos]
        if len(alive) == 0:
            break
    out_tok, out_lp = out_tok.tolist(), out_lp.tolist()
    result = []
    for ids, lps in zip(out_tok, out_lp):
        n = sum(i >= 0 for i in ids)
        result.append((ids[:n], lps[:n]))
    return result


@torch.no_grad()
def rollout_batch(policy, codec: Codec, envs: list[FileSystemEnv], *, temperature: float = 1.0,
                  forced: list[list[str]] | None = None, max_action_tokens: int = 6, max_len: int = 384,
                  device: str = "cpu", version: int = 0, step_fn=None) -> list[Trajectory]:
    """Run every env to the end. ``step_fn(envs, actions) -> [(obs, reward, done)]`` lets a worker pool run the
    tools; by default they run inline, one after the other."""
    step_fn = step_fn or (lambda es, acts: [e.step(a) for e, a in zip(es, acts)])
    trajs = [new_trajectory(env, prompt_ids(codec, env.reset()), version) for env in envs]
    pending = [list(f) for f in forced] if forced else [[] for _ in envs]
    active = list(range(len(envs)))
    while active:
        # 1. every episode waiting for an action shares one batched generation
        gen = [i for i in active if not pending[i]]
        sampled = dict(zip(gen, sample_actions(policy, [trajs[i].tokens for i in gen], codec, temperature,
                                               max_action_tokens, device))) if gen else {}
        # 2. record the actions as tokens
        texts = []
        for i in active:
            tr, step = trajs[i], len(envs[i].history)
            tr.state_positions.append(len(tr.tokens) - 1)
            if pending[i]:                                 # an injected mistake: played, never trained on
                texts.append(pending[i].pop(0))
                append(tr, codec.encode(texts[-1]) + [codec.eos], None, step)
            else:
                ids, lps = sampled[i]
                append(tr, ids, lps, step)
                texts.append(codec.decode([t for t in ids if t != codec.eos]))
        # 3. run the tools, then append what they answered
        results = step_fn([envs[i] for i in active], texts)
        still = []
        for i, (obs, _, done) in zip(active, results):
            tr, env = trajs[i], envs[i]
            env.history[-1].forced = i not in sampled
            if done or len(tr.tokens) + max_action_tokens + OBS_BUDGET > max_len:
                continue
            append(tr, observation_ids(codec, obs), None, len(env.history) - 1)
            still.append(i)
        active = still
    return [finish(tr, env) for tr, env in zip(trajs, envs)]
