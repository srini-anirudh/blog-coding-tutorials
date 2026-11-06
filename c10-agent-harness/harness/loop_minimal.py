"""The smallest agent loop that works, kept as the reference the full Agent grows from."""

from __future__ import annotations

from typing import Callable


def run_minimal(model, tools: dict[str, Callable[..., str]], task: str) -> list[dict]:
    context = [{"role": "user", "content": task}]
    while True:
        response = model.complete(context, tools=[])
        context.append(response.to_message().to_dict())
        if response.final:
            break
        for call in response.tool_calls:
            result = tools[call.name](**call.arguments)
            context.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return context
