"""Code companion for the post "Inside an AI Agent Harness".

Each module makes one claim of the post run on a small scale:

    prefix_cache   a block-hashed prefix cache that counts recomputed and billed tokens
    loop_cost      the quadratic loop, caching, append-only, tool order, compaction, tool tax
    kv_reuse       the same prefix reuse on a real transformer KV cache, checked for exactness
    code_action    one tool call per item against one code action, on a real temp repo
    edit_cost      the geometric retry cost of an edit format, using C10's patch appliers
    observation    five ways to put a large tool output into the context
    permissions    the six-step approval ladder and the vigilance-decay model
    stopping       geometric turn length, caps, and a termination predicate

The C10 tutorial (`c10-agent-harness`) builds the harness itself. This lab imports its
edit appliers, token estimate and truncation policy instead of rebuilding them.
"""

from __future__ import annotations

import sys
from pathlib import Path

C10 = Path(__file__).resolve().parents[3] / "c10-agent-harness"
if C10.is_dir() and str(C10) not in sys.path:
    sys.path.insert(0, str(C10))
