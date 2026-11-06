"""The production loop: the minimal loop plus every guard rail it was missing.

Per step:

    build context (compact if over budget)
      -> call model (retry transient failures)
      -> parse errors?        feed them back, count a protocol error
      -> no tool calls?       done
      -> for each call:  validate -> permission check -> execute in sandbox
                         -> truncate the observation -> log -> append to history

Every arrow writes an event, so the JSONL log is a complete record of the run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .compaction import Summarizer, extractive_summary, plan_compaction
from .context import AgentState, Context, ContextBuilder, repo_section
from .events import EventLog
from .fs_tools import FileTracker
from .git_tools import create_checkpoint, is_repo
from .messages import Message, ModelResponse, ToolCall
from .models.base import ModelClient, ModelError, TransientModelError
from .parsing import format_call_errors
from .permissions import PermissionPolicy, Risk
from .planning import Plan
from .sandbox import LocalSandbox, Sandbox
from .tools import ToolContext, ToolOutput, ToolRegistry
from .tracing import Tracer
from .truncation import truncate_output


@dataclass
class AgentConfig:
    max_steps: int = 30
    max_protocol_errors: int = 4      # consecutive malformed or invalid calls before giving up
    model_retries: int = 3
    retry_backoff_s: float = 0.5
    context_budget: int = 16_000      # estimated tokens; compaction runs above this
    keep_recent_turns: int = 4        # turns kept verbatim when compacting (fewer if needed)
    compact_to: float = 0.7           # after compaction, aim for this fraction of the budget
    max_observation_lines: int = 150
    checkpoint_on_start: bool = True


@dataclass
class AgentResult:
    status: str                       # "done" | "max_steps" | "protocol_errors" | "model_error"
    final_text: str
    steps: int
    trace: dict = field(default_factory=dict)


class Agent:
    def __init__(self, model: ModelClient, tools: ToolRegistry, workspace: str | Path, *,
                 sandbox: Sandbox | None = None, policy: PermissionPolicy | None = None,
                 log: EventLog | None = None, context: ContextBuilder | None = None,
                 tracer: Tracer | None = None, config: AgentConfig | None = None,
                 summarizer: Summarizer = extractive_summary):
        self.model = model
        self.tools = tools
        self.workspace = Path(workspace).resolve()
        self.sandbox = sandbox or LocalSandbox(self.workspace)
        self.policy = policy or PermissionPolicy()
        self.log = log or EventLog()
        self.builder = context or ContextBuilder(sections={"repo": repo_section()})
        self.tracer = tracer or Tracer()
        self.config = config or AgentConfig()
        self.summarizer = summarizer
        self.ctx = ToolContext(self.workspace, self.sandbox, FileTracker(), Plan())
        self.history: list[tuple[int, Message]] = []   # (seq of the event that produced it, message)
        self.state = AgentState("", self.workspace, plan=self.ctx.plan)

    # ------------------------------------------------------------ the loop

    def run(self, task: str) -> AgentResult:
        self.state.task = task
        self._append_message(self.log.append("UserMessage", 0, content=task).seq, Message("user", task))
        if self.config.checkpoint_on_start and is_repo(self.workspace):
            sha = create_checkpoint(self.workspace, "start of run")
            self.log.append("Checkpoint", 0, sha=sha, label="start of run")

        protocol_errors = 0
        for step in range(1, self.config.max_steps + 1):
            self.state.step = step
            context = self._build_context(step)
            try:
                response = self._call_model(context, step)
            except ModelError as e:
                self.log.append("Error", step, kind="model", message=str(e))
                return self._finish("model_error", str(e), step)
            ev = self.log.append("ModelResponse", step, text=response.text,
                                 tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments}
                                             for c in response.tool_calls],
                                 parse_errors=response.parse_errors,
                                 usage={"prompt": response.usage.prompt_tokens,
                                        "completion": response.usage.completion_tokens},
                                 context=context.breakdown)
            self._append_message(ev.seq, response.to_message())

            if response.final:
                self.log.append("Final", step, text=response.text)
                return self._finish("done", response.text, step)

            step_had_protocol_error = bool(response.parse_errors)
            for call in response.tool_calls:
                step_had_protocol_error |= self._run_tool(call, step)
            if response.parse_errors:
                feedback = format_call_errors(None, response.parse_errors)
                ev = self.log.append("UserMessage", step, content=feedback, kind="feedback")
                self._append_message(ev.seq, Message("user", feedback))

            protocol_errors = protocol_errors + 1 if step_had_protocol_error else 0
            if protocol_errors >= self.config.max_protocol_errors:
                return self._finish("protocol_errors", "too many malformed tool calls in a row", step)
        return self._finish("max_steps", "step budget exhausted", self.config.max_steps)

    # ------------------------------------------------------------ pieces

    def _build_context(self, step: int) -> Context:
        messages = [m for _, m in self.history]
        context = self.builder.build(self.state, messages, self.tools.schemas())
        if context.tokens <= self.config.context_budget:
            return context
        # Compact down to a target well under the budget, not just under it; otherwise
        # every following turn crosses the line again and pays for another compaction.
        target = int(self.config.context_budget * self.config.compact_to)
        plan = after = None
        for keep in range(self.config.keep_recent_turns, 0, -1):
            plan = plan_compaction(messages, keep, self.summarizer)
            if plan is None:
                continue
            summary_msg = Message("user", plan.summary)
            after = self.builder.build(self.state, [summary_msg] + messages[plan.cut:], self.tools.schemas())
            if after.tokens <= target:
                break
        if plan is None or after is None:
            return context  # nothing old enough to drop; the model sees an over-budget context
        kept_from = self.history[plan.cut][0]
        ev = self.log.append("Compaction", step, summary=plan.summary, kept_from_seq=kept_from,
                             dropped_messages=plan.dropped, tokens_before=context.tokens,
                             tokens_after=after.tokens)
        self.history = [(ev.seq, summary_msg)] + self.history[plan.cut:]
        return after

    def _call_model(self, context: Context, step: int) -> ModelResponse:
        with self.tracer.span("model", self.model.name, step) as span:
            for attempt in range(self.config.model_retries + 1):
                try:
                    response = self.model.complete(context.messages, context.tools)
                    break
                except TransientModelError as e:
                    if attempt == self.config.model_retries:
                        raise
                    span.retries += 1
                    self.log.append("Error", step, kind="transient", message=str(e), attempt=attempt + 1)
                    time.sleep(self.config.retry_backoff_s * 2 ** attempt)
            span.prompt_tokens = response.usage.prompt_tokens
            span.completion_tokens = response.usage.completion_tokens
        return response

    def _run_tool(self, call: ToolCall, step: int) -> bool:
        """Validate, authorise, execute and record one call. Returns True on a protocol error."""
        self.log.append("ToolCall", step, id=call.id, name=call.name, arguments=call.arguments)
        errors = self.tools.validate(call)
        if errors:
            out = ToolOutput(format_call_errors(call, errors), is_error=True)
        else:
            risk = self.tools[call.name].risk_for(call.arguments)
            decision = self.policy.check(call.name, call.arguments, risk)
            if risk > Risk.READ or not decision.allowed:
                self.log.append("Approval", step, id=call.id, name=call.name, risk=risk.name,
                                allowed=decision.allowed, asked=decision.asked, reason=decision.reason)
            if not decision.allowed:
                out = ToolOutput(f"Permission denied for {call.name} ({risk.name}): {decision.reason}. "
                                 "Choose a different approach.", is_error=True)
            else:
                with self.tracer.span("tool", call.name, step) as span:
                    out = self.tools.execute(call, self.ctx)
                    span.ok = not out.is_error
        content = self._observe(out.content)
        result = self.log.append("ToolResult", step, id=call.id, name=call.name, content=content,
                                 is_error=out.is_error)
        if out.kind == "patch":
            self.log.append("Patch", step, id=call.id, **out.data)
        elif out.kind == "test":
            d = out.data
            self.log.append("TestResult", step, id=call.id, check=d["check"], status=d["status"],
                            passed=d["passed"], failed=d["failed"], errors=d["errors"])
        self._append_message(result.seq, Message("tool", content, tool_call_id=call.id, name=call.name))
        return bool(errors)

    def _observe(self, text: str) -> str:
        """Last line of defence: no single observation may exceed the line budget."""
        return truncate_output(text, max_lines=self.config.max_observation_lines).text

    def _append_message(self, seq: int, msg: Message) -> None:
        # Each message remembers the event that produced it, so a compaction recorded
        # as "keep everything from seq N" replays to exactly this history.
        self.history.append((seq, msg))

    def _finish(self, status: str, text: str, steps: int) -> AgentResult:
        return AgentResult(status, text, steps, self.tracer.summary())
