"""harness: a coding-agent harness built from an 8-line loop up.

Public API (stable; tutorial C11 extends it):

    Agent, AgentConfig, AgentResult        the loop
    ToolRegistry, tool, ToolContext,       tools and their schemas
    ToolOutput, ToolError
    EventLog, Event                        the append-only JSONL record
    ContextBuilder, AgentState             what the model sees each turn
    Message, ToolCall, ModelResponse       the data on the wire
    ModelClient, ScriptedModel,            model clients (HFLocalModel is lazy: it imports torch)
    HFLocalModel, OpenAICompatModel
    PermissionPolicy, Risk                 the permission layer
    LocalSandbox, DockerSandbox, Limits    where commands run
    coding_tools                           the default toolset
"""

from .agent import Agent, AgentConfig, AgentResult
from .context import AgentState, ContextBuilder, repo_section
from .events import Event, EventLog
from .messages import Message, ModelResponse, ToolCall, Usage
from .models import ModelClient, OpenAICompatModel, ScriptedModel, call
from .permissions import PermissionPolicy, Risk, classify_command
from .sandbox import DockerSandbox, Limits, LocalSandbox
from .tools import ToolContext, ToolError, ToolOutput, ToolRegistry, tool
from .toolset import coding_tools
from .tracing import Tracer

__all__ = [
    "Agent", "AgentConfig", "AgentResult", "AgentState", "ContextBuilder", "repo_section",
    "Event", "EventLog", "Message", "ModelResponse", "ToolCall", "Usage",
    "ModelClient", "OpenAICompatModel", "ScriptedModel", "HFLocalModel", "call",
    "PermissionPolicy", "Risk", "classify_command", "DockerSandbox", "Limits", "LocalSandbox",
    "ToolContext", "ToolError", "ToolOutput", "ToolRegistry", "tool", "coding_tools", "Tracer",
]


def __getattr__(name: str):
    if name == "HFLocalModel":
        from .models.hf_local import HFLocalModel
        return HFLocalModel
    raise AttributeError(name)
