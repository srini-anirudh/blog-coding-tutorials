"""organs: the parts of a Transformer block that changed between 2017 and 2026, small enough to run on a CPU.

Companion code for https://srini-anirudh.github.io/blogs/what-changed-inside-the-transformer/

The building blocks that C3 (attention) and C4 (a modern LLM) already build are imported from there:
RMSNorm, SwiGLU, RoPE and the Switch load-balancing loss come from `tinyllm`, masks from `attnzoo`.
This package adds what the post argues: how the pieces behave, and the arithmetic behind each change.
"""

from . import _paths  # noqa: F401  (puts c03-attention and c04-llm on sys.path)
