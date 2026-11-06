"""A toy closed loop at 50 Hz: chunk size, inference latency and the reaction to a surprise (the post's Figure 3).

An object moves smoothly and is bumped at 2.2 s. The policy observes the object's position and velocity and predicts
H future hand positions by extrapolation. Each call adds one random offset, as a sampled generative policy would.

Three execution modes:
  sync   : when a chunk runs out, call the policy and hold still until it answers.
  async  : call the policy `latency` before the chunk runs out, so the next chunk is ready in time. The part of the
           new chunk that covers the waiting time is dropped (it would act on the past).
  blend  : async, but call `overlap` steps earlier still, and cross-fade old and new plans over the overlap. This is a
           simple stand-in for real-time chunking (RTC), which instead inpaints the new chunk; it is not RTC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class LoopConfig:
    hz: float = 50.0
    seconds: float = 4.0
    bump_time: float = 2.2
    bump: float = 0.3
    offset_std: float = 0.01


def object_path(cfg: LoopConfig) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity. The velocity is that of the smooth motion: the bump is a jump, not a speed."""
    t = np.arange(int(cfg.seconds * cfg.hz)) / cfg.hz
    w = 2 * math.pi * 0.25
    return 0.3 * np.sin(w * t) + np.where(t >= cfg.bump_time, cfg.bump, 0.0), 0.3 * w * np.cos(w * t)


def simulate(H: int, latency_ms: float, mode: str, cfg: LoopConfig = LoopConfig(), seed: int = 0,
             overlap: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    obj, vel = object_path(cfg)
    T, dt = len(obj), 1.0 / cfg.hz
    L = int(math.ceil(latency_ms / 1000 * cfg.hz - 1e-9))  # steps until an answer arrives
    early = L + (overlap if mode == "blend" else 0)
    plan = np.full(T + H + L + overlap + 1, np.nan)  # planned hand position per absolute step
    seen = np.full(len(plan), -1)  # the step whose observation produced each planned position
    hand, h = np.zeros(T), 0.0
    pending = None  # (ready_step, chunk_start_step, chunk)
    calls, paused, boundaries = 0, 0, []
    src = np.full(T, -1)  # which observation the executed command came from

    def call(k):
        start = k + L  # first step the answer can drive
        steps = np.arange(start, start + H)
        return (start, start, obj[k] + vel[k] * (steps - k) * dt + rng.normal(0, cfg.offset_std), k)

    for k in range(T):
        planned_until = np.where(~np.isnan(plan[k:]))[0]
        remaining = int(planned_until.max()) + 1 if planned_until.size else 0
        if pending is None:
            if (mode == "sync" and remaining == 0) or (mode != "sync" and remaining <= early):
                pending = call(k)
                calls += 1
        if pending is not None and k >= pending[0]:
            _, start, chunk, obs_k = pending
            idx = np.arange(start, start + H)
            keep = idx >= k  # parts that cover time already gone are dropped
            new = chunk[keep]
            idx = idx[keep]
            old = plan[idx]
            if mode == "blend":
                w = np.clip(1 - (idx - k + 1) / (overlap + 1), 0, 1)  # weight on the old plan, fading out
                new = np.where(np.isnan(old), new, w * np.nan_to_num(old) + (1 - w) * new)
            plan[idx] = new
            seen[idx] = obs_k
            plan[idx.max() + 1:] = np.nan
            boundaries.append(k)
            pending = None
        if np.isnan(plan[k]):
            paused += 1  # nothing to execute: hold still
            src[k] = -1
        else:
            h = plan[k]
            src[k] = seen[k]
        hand[k] = h

    bump_k = int(cfg.bump_time * cfg.hz)
    # Reaction: the first executed command that was computed from an observation taken after the bump.
    after = np.where(src[bump_k:] >= bump_k)[0]
    steps = np.abs(np.diff(hand))
    b = [i for i in boundaries if 0 < i < T]
    return {"calls_per_s": calls / cfg.seconds, "paused_frac": paused / T,
            "reaction_ms": float(after[0] * dt * 1000) if after.size else float("nan"),
            "jump_at_switch": float(steps[np.array(b) - 1].mean()) if b else 0.0,
            "step_elsewhere": float(np.delete(steps, np.array(b) - 1).mean()) if b else float(steps.mean()),
            "track_err": float(np.abs(hand - obj)[5:].mean())}


def sweep(Hs, latencies, modes, seeds: int = 20, cfg: LoopConfig = LoopConfig()) -> list[dict]:
    """Mean of each metric over `seeds` runs. Seed s bumps the object at bump_time - 0.6 s + 3s steps, so 20 seeds span 1.2 s."""
    rows = []
    for lat in latencies:
        for mode in modes:
            for H in Hs:
                # Reaction depends on where the bump falls inside a chunk, so each seed moves it by three steps.
                r = [simulate(H, lat, mode, replace(cfg, bump_time=cfg.bump_time - 0.6 + 3 * s / cfg.hz), seed=s)
                     for s in range(seeds)]
                rows.append({"latency_ms": lat, "mode": mode, "H": H,
                             **{k: float(np.nanmean([x[k] for x in r])) for k in r[0]}})
    return rows
