"""Model clients. All implement `complete(messages, tools) -> ModelResponse`."""

from .base import ModelClient, ModelError, TransientModelError
from .openai_compat import OpenAICompatModel
from .scripted import ScriptedModel, ScriptExhausted, call

__all__ = ["ModelClient", "ModelError", "TransientModelError", "ScriptedModel", "ScriptExhausted",
           "call", "OpenAICompatModel", "HFLocalModel"]


def __getattr__(name: str):
    # Import torch/transformers only when the HF client is actually requested.
    if name == "HFLocalModel":
        from .hf_local import HFLocalModel
        return HFLocalModel
    raise AttributeError(name)
