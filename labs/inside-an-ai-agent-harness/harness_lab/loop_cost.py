"""What one agent turn costs in input tokens, run through a real prefix cache.

Every step re-sends the whole context. Tokens here are fresh integer ids, so two
blocks match only when they are the same block. Each function plays one turn and
returns counts. The closed forms from the post sit next to them so tests can check
that the simulation and the algebra agree.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .prefix_cache import PrefixCache, Request


class Fresh:
    """Hands out token ids that were never used before."""

    def __init__(self) -> None:
        self.next = 0

    def __call__(self, n: int) -> list[int]:
        out = list(range(self.next, self.next + n))
        self.next += n
        return out


@dataclass
class Turn:
    requests: list[Request] = field(default_factory=list)
    compactions: int = 0
    output_tokens: int = 0     # tokens the model wrote for compaction summaries

    @property
    def sent(self) -> int:
        return sum(q.total for q in self.requests)

    @property
    def computed(self) -> int:
        return sum(q.computed for q in self.requests)

    @property
    def cached(self) -> int:
        return sum(q.cached for q in self.requests)

    def billed(self, r: float) -> float:
        return sum(q.billed(r) for q in self.requests)


# ------------------------------------------------------------------ closed forms (post, section 3)

def total_input(K: int, P0: int, delta: int) -> int:
    """T = K P0 + delta K (K + 1) / 2: the input processed with no cache."""
    return K * P0 + delta * K * (K + 1) // 2


def effective_input(K: int, P0: int, delta: int, h: float, r: float) -> float:
    """T_eff: a fraction h of calls hit the cache, and cached tokens cost r."""
    return sum(h * (r * (P0 + (k - 1) * delta) + delta) + (1 - h) * (P0 + k * delta)
               for k in range(1, K + 1))


# ------------------------------------------------------------------ the plain loop

def plain_turn(K: int, P0: int, delta: int, h: float = 1.0, block_size: int = 16, seed: int = 0,
               warm: bool = True) -> Turn:
    """K steps. Each step adds delta tokens (model output plus tool result).

    With warm=True the fixed prefix P0 was seen by an earlier turn, so it starts out
    cached. A call misses the cache with probability 1 - h, as when a request lands
    on a server that does not hold the prefix.
    """
    fresh, rng = Fresh(), random.Random(seed)
    cache = PrefixCache(block_size)
    prefix = fresh(P0)
    if warm:
        cache.insert(prefix)
    ctx = prefix + fresh(delta)          # the user's message
    turn = Turn()
    for _ in range(K):
        turn.requests.append(cache.request(ctx, use_cache=rng.random() < h))
        ctx = ctx + fresh(delta)
    return turn


# ------------------------------------------------------------------ append versus edit (section 5)

DEFAULT_BLOCKS = {"system": 2000, "tools": 4000, "model": 496, "permissions": 304, "developer": 496,
                  "project": 4000, "skills": 496, "environment": 208}


def state_change_turn(lane: str, K: int = 40, change_at: int = 20, delta: int = 1024,
                      blocks: dict[str, int] | None = None, block_size: int = 16) -> list[Request]:
    """The working directory changes before step `change_at`.

    lane "append": add a new environment message at the end, as Codex does.
    lane "edit":   rewrite the environment block where it sits, near the front.
    """
    blocks = dict(blocks or DEFAULT_BLOCKS)
    fresh, cache = Fresh(), PrefixCache(block_size)
    prefix = {name: fresh(n) for name, n in blocks.items()}
    history = fresh(delta)
    out = []
    for k in range(1, K + 1):
        if k == change_at:
            if lane == "append":
                history = history + fresh(blocks["environment"])
            elif lane == "edit":
                prefix["environment"] = fresh(blocks["environment"])
            else:
                raise ValueError(lane)
        ctx = [t for toks in prefix.values() for t in toks] + history
        out.append(cache.request(ctx))
        history = history + fresh(delta)
    return out


def edit_item_cost(n_items: int = 40, item_tokens: int = 128, edited: int = 6, block_size: int = 16) -> Request:
    """Send a list of n items, then send it again with item `edited` (1-based) changed."""
    fresh, cache = Fresh(), PrefixCache(block_size)
    items = [fresh(item_tokens) for _ in range(n_items)]
    cache.request([t for it in items for t in it])
    items[edited - 1] = fresh(item_tokens)
    return cache.request([t for it in items for t in it])


# ------------------------------------------------------------------ tool order (section 5)

def tool_order_turn(orders: list[list[str]], tool_tokens: int = 160, delta: int = 512,
                    system: int = 2000, block_size: int = 16) -> list[Request]:
    """One request per entry of `orders`. Each lists the tool schemas in that order."""
    fresh, cache = Fresh(), PrefixCache(block_size)
    sys_toks = fresh(system)
    schemas = {name: fresh(tool_tokens) for name in sorted(orders[0])}
    history = fresh(delta)
    out = []
    for order in orders:
        ctx = sys_toks + [t for name in order for t in schemas[name]] + history
        out.append(cache.request(ctx))
        history = history + fresh(delta)
    return out


# ------------------------------------------------------------------ compaction (section 16)

def compaction_turn(K: int, P0: int, delta: int, threshold: int | None, summary_tokens: int,
                    block_size: int = 16) -> Turn:
    """Before a step, if the context is over `threshold`, compact it.

    Compaction is one extra request that reads the whole context and writes a
    summary. The new context is the fixed prefix plus that summary. The prefix keeps
    its cache. The summary is new text, so everything after the prefix is recomputed.
    """
    fresh, cache = Fresh(), PrefixCache(block_size)
    prefix = fresh(P0)
    cache.insert(prefix)
    ctx = prefix + fresh(delta)
    turn = Turn()
    for _ in range(K):
        if threshold is not None and len(ctx) > threshold:
            turn.requests.append(cache.request(ctx))   # the summarisation call
            turn.compactions += 1
            turn.output_tokens += summary_tokens
            ctx = prefix + fresh(summary_tokens)
        turn.requests.append(cache.request(ctx))
        ctx = ctx + fresh(delta)
    return turn


# ------------------------------------------------------------------ tool-context tax (section 17)

def tool_tax(K: int, M: int, tau: int) -> int:
    """T_tools = K M tau: schema tokens sent across a turn when every schema is loaded up front."""
    return K * M * tau


def tool_loading_turn(mode: str, K: int = 40, M: int = 300, tau: int = 120, used: int = 2,
                      P0: int = 2000, delta: int = 512, block_size: int = 8) -> tuple[Turn, int]:
    """Play a turn with all M schemas up front ("eager"), or one search tool that
    loads `used` schemas into the history at step 2 ("deferred").

    Returns the turn and the number of schema tokens sent across it.
    """
    fresh, cache = Fresh(), PrefixCache(block_size)
    base = fresh(P0)
    n_up_front = M if mode == "eager" else 1 if mode == "deferred" else None
    if n_up_front is None:
        raise ValueError(mode)
    schemas = fresh(n_up_front * tau)
    cache.insert(base + schemas)
    history = fresh(delta)
    turn, schema_sent, loaded = Turn(), 0, 0
    for k in range(1, K + 1):
        if mode == "deferred" and k == 2:
            history = history + fresh(used * tau)     # the search result: two schemas
            loaded = used * tau
        turn.requests.append(cache.request(base + schemas + history))
        schema_sent += len(schemas) + loaded
        history = history + fresh(delta)
    return turn, schema_sent
