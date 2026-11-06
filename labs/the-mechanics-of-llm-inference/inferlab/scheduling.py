"""Static against continuous batching, simulated in units of one decode step.

The model is deliberately crude: every decode step costs 1, a prefill of N_p tokens costs max(1, N_p / 340) steps
(the post's figure 15 uses the same rule), and a prefill blocks every running sequence while it runs. It is a
slot-accounting model, not a timing model. Its job is to show where the slot-steps go.
"""

from __future__ import annotations

from dataclasses import dataclass

# The eight requests of the post's figure 15: (arrival step, prompt tokens, output tokens).
POST_REQUESTS = [(0, 512, 40), (0, 2048, 300), (1, 128, 20), (3, 8192, 60),
                 (5, 256, 500), (8, 1024, 80), (12, 512, 150), (15, 4096, 30)]


def prefill_cost(n_prompt: int) -> float:
    return max(1.0, n_prompt / 340)


@dataclass
class ScheduleResult:
    name: str
    steps: float  # wall time in decode-step units
    useful: int  # slot-steps that produced a token
    held: float  # slots x decode steps: the batch capacity that was paid for
    finish: dict[int, float]  # request index -> completion time
    stall: float = 0.0  # running-sequence x steps spent waiting on someone else's prefill

    @property
    def utilisation(self) -> float:
        return self.useful / self.held

    def mean_latency(self, requests) -> float:
        return sum(self.finish[i] - requests[i][0] for i in self.finish) / len(self.finish)


def static_batching(requests, slots: int) -> ScheduleResult:
    """Take requests in arrival order, `slots` at a time. A batch runs until its longest member is done."""
    t, useful, held, finish = 0.0, 0, 0.0, {}
    for b0 in range(0, len(requests), slots):
        batch = list(range(b0, min(b0 + slots, len(requests))))
        t = max(t, max(requests[i][0] for i in batch))  # wait until the whole batch has arrived
        t += sum(prefill_cost(requests[i][1]) for i in batch)
        longest = max(requests[i][2] for i in batch)
        for i in batch:
            finish[i] = t + longest  # nobody returns before the slowest
        useful += sum(requests[i][2] for i in batch)
        held += slots * longest
        t += longest
    return ScheduleResult(f"static, {slots} slots", t, useful, held, finish)


def continuous_batching(requests, slots: int) -> ScheduleResult:
    """Iteration-level scheduling: after every step, finished sequences leave and waiting ones take their slot."""
    t, useful, held, stall, finish = 0.0, 0, 0.0, 0.0, {}
    waiting = list(range(len(requests)))
    running: dict[int, int] = {}  # request -> tokens still to generate
    while waiting or running:
        while waiting and len(running) < slots and requests[waiting[0]][0] <= t:
            i = waiting.pop(0)
            t += prefill_cost(requests[i][1])  # the prefill stalls everyone already running
            stall += len(running) * prefill_cost(requests[i][1])
            running[i] = requests[i][2]
        if not running:
            t = requests[waiting[0]][0]
            continue
        t += 1
        useful += len(running)
        held += slots  # every slot is reserved for this step, busy or not
        for i in list(running):
            running[i] -= 1
            if running[i] == 0:
                finish[i] = t
                del running[i]
    return ScheduleResult(f"continuous, {slots} slots", t, useful, held, finish, stall)
