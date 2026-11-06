"""An agent is a fold over contexts.

    x_{k+1} = x_k + y_k + e(Exec(Parse(y_k))),   y_k ~ p_theta(. | x_k)

The model is a pure function from a context to a span of text. Parse, Exec and e live here, in the
harness. Nothing is remembered between steps except what is written into the context.
C10 (c10-agent-harness) builds the full production version of this loop; this file is the bare fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Step:
    context_len: int  # characters in x_k before this step
    span: str  # y_k, what the model emitted
    call: dict | None  # Parse(y_k)
    observation: str | None  # e(Exec(...)), what the harness wrote back


@dataclass
class Trace:
    context: str
    steps: list[Step] = field(default_factory=list)


def fold(model: Callable[[str], str], x0: str, parse: Callable[[str], dict | None],
         execute: Callable[[dict], str], encode: Callable[[str], str], max_steps: int = 8) -> Trace:
    """Run the loop until the model emits a span that is not a call, or max_steps is reached."""
    tr = Trace(context=x0)
    for _ in range(max_steps):
        span = model(tr.context)  # one forward pass (or one decode), nothing more
        call = parse(span)
        if call is None:  # not a call: the turn ends, the span is the answer
            tr.steps.append(Step(len(tr.context), span, None, None))
            tr.context += span
            break
        try:
            obs = encode(execute(call))
        except Exception as err:  # an error is also just text the model will read next
            obs = encode(f"error: {err}")
        tr.steps.append(Step(len(tr.context), span, call, obs))
        tr.context = tr.context + span + obs  # the whole state of the agent
    return tr
