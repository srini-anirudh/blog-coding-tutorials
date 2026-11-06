"""Section "Four kinds of state": a context compaction erases the record of a write, and the agent writes again.

    python labs/the-making-of-an-ai-agent/scripts/05_state_layers.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcraft.common import table, write_results
from agentcraft.state_layers import HarnessConfig, make_todo, run_episode

VARIANTS = [
    ("naive harness (context only)", HarnessConfig()),
    ("+ persistent write log re-injected after compaction", HarnessConfig(write_log=True)),
    ("+ policy lists events before each write", HarnessConfig(check_world=True)),
    ("+ idempotent create in the tool", HarnessConfig(idempotent=True)),
    ("no compaction (context budget 1000)", HarnessConfig(context_budget=1000)),
    ("token expires after 15 calls, no refresh", HarnessConfig(token_ttl=15, write_log=True)),
    ("token expires after 15 calls, harness refreshes", HarnessConfig(token_ttl=15, write_log=True,
                                                                      refresh_token=True)),
]


def main():
    episodes = 50
    todos = [make_todo(random.Random(seed)) for seed in range(episodes)]
    rows, data = [], []
    for name, cfg in VARIANTS:
        rs = [run_episode(t, cfg) for t in todos]
        r = {"variant": name, "duplicate_events": sum(x.duplicates for x in rs) / episodes,
             "missing_events": sum(x.missing for x in rs) / episodes, "tool_calls": sum(x.calls for x in rs) / episodes,
             "compactions": sum(x.compactions for x in rs) / episodes,
             "finished": sum(x.finished for x in rs) / episodes, "tool_errors": sum(x.errors for x in rs) / episodes}
        data.append(r)
        rows.append([name, r["duplicate_events"], r["missing_events"], r["tool_calls"], r["compactions"],
                     r["tool_errors"], r["finished"]])
    first = run_episode(todos[0], VARIANTS[0][1])
    text = (f"{episodes} episodes. Each asks for 6 calendar events with 1 to 3 email reads after each. Context budget "
            "12 lines; compaction keeps the task, a summary of emails read, and the last 6 lines.\n"
            "Per-episode means. The policy is scripted: it trusts its transcript (see agentcraft/state_layers.py).\n\n"
            + table(rows, ["harness", "duplicate events", "missing events", "tool calls", "compactions",
                           "tool errors", "finished"])
            + "\n\nFinal context of episode 0 under the naive harness:\n" + "\n".join(
                line[:100] for line in first.context))
    print(text)
    write_results("05_state_layers", text, {"episodes": episodes, "rows": data})


if __name__ == "__main__":
    main()
