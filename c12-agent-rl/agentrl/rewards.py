"""Per-step reward functions. Each maps a trajectory to one reward per step.

    outcome    1[success] on the last step, 0 elsewhere. The reward we actually care about.
    activity   +0.2 for every tool call that did not error, plus the outcome. Looks reasonable ("reward
               productive tool use"). It is a deliberate reward hack: looping reads until the step budget runs
               out pays more than solving the task.
    potential  outcome + gamma * Phi(s_{t+1}) - Phi(s_t), with Phi = fraction of milestones reached (Ng et al.,
               1999). Phi is 0 at the start and at the end, so the shaping terms telescope: the episode's total
               shaped return equals its outcome. It changes *when* credit arrives, never *what* is optimal.
"""

from __future__ import annotations

from .trajectory import Trajectory


def outcome(traj: Trajectory) -> list[float]:
    r = [0.0] * traj.n_steps
    if r:
        r[-1] = traj.outcome
    return r


def activity(traj: Trajectory, bonus: float = 0.2) -> list[float]:
    r = [bonus if (not s.error and s.action != "submit") else 0.0 for s in traj.steps]
    if r:
        r[-1] += traj.outcome
    return r


def potential(traj: Trajectory, gamma: float = 1.0) -> list[float]:
    r = outcome(traj)
    phi_prev = 0.0                       # Phi(s_0) = 0: nothing found or edited yet
    for t, s in enumerate(traj.steps):
        phi_next = s.progress            # the env sets Phi = 0 once the episode is over
        if t == traj.n_steps - 1 and not s.reward and not traj.submitted:
            phi_next = 0.0               # context-truncated episodes end too
        r[t] += gamma * phi_next - phi_prev
        phi_prev = phi_next
    return r


REWARDS = {"outcome": outcome, "activity": activity, "potential": potential}
