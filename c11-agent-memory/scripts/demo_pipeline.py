"""Walk the memory pipeline on a small script and print every intermediate state.

    python c11-agent-memory/scripts/demo_pipeline.py

Sections: writer decisions and fact operations; the embedding matrix and retrieval
terms; consolidation; the packed context. Output is saved to results/demo_pipeline.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from agentmem import AgentMemory, Episode, MemoryEvent, TaskState, count_tokens  # noqa: E402
from agentmem.records import reset_ids  # noqa: E402

EVENTS = [
    (1.1, "user", "The deploy region for atlas is eu-west-1."),
    (1.2, "user", "Thanks, that helps."),
    (1.3, "tool", "12:01:33 INFO api request id=9f3a2b77c1d0 status=200 latency=41ms"),
    (1.4, "user", "The owner for atlas is priya."),
    (1.5, "user", "The test command for atlas is `pytest -x`."),
    (2.1, "user", "Remember: never deploy on Fridays."),
    (2.2, "user", "The database for cobalt is postgres."),
    (3.1, "user", "Atlas's deploy region is eu-west-1."),
    (4.1, "user", "Heads up: we moved the deploy region for atlas to us-east-2."),
    (4.2, "user", "I think the deploy region for atlas might be ap-south-1, not sure."),
    (4.3, "user", "Just for today, the owner for atlas is marco."),
    (5.1, "user", "Forget the test command for atlas, we no longer use it."),
]


def main() -> None:
    reset_ids()
    mem = AgentMemory()
    print("WRITER AND FACT STORE")
    print(f"{'t':>4} {'source':<5} {'nov':>5} {'stab':>5} {'imp':>5} {'conf':>5} {'scope':<8} {'score':>5} {'store':<6} op          event")
    for t, source, text in EVENTS:
        ents = [e for e in ("atlas", "cobalt") if e in text.lower()]
        d, ops = mem.observe(MemoryEvent(text, source, t, entities=ents))
        s = d.scores
        op = ",".join(o.value if hasattr(o, "value") else str(o) for o, _ in ops) or "-"
        print(f"{t:>4} {source:<5} {s.novelty:>5.2f} {s.stability:>5.2f} {s.importance:>5.2f} {s.confidence:>5.2f} "
              f"{s.scope:<8} {d.score:>5.2f} {str(d.store):<6} {op:<11} {text[:52]}")

    print("\nFACT HISTORY for (atlas, deploy region)")
    for f in mem.facts.history("atlas", "deploy region"):
        print(f"  {f.value:<12} from t={f.valid_from} to t={f.valid_to} status={f.status} support={f.support} conf={f.confidence:.2f}")
    print("  as_of(t=2.0) ->", mem.facts.as_of("atlas", "deploy region", 2.0).value)
    print("  as_of(t=4.5) ->", mem.facts.as_of("atlas", "deploy region", 4.5).value)
    print("  conflicts logged:", [c.evidence for c in mem.facts.conflicts()])
    print("  session notes   :", mem.session_notes)

    idx = mem.retriever.index
    print("\nTHE VECTOR INDEX")
    print(f"  matrix {tuple(idx.matrix.shape)}  alive {int(idx.alive.sum())}  high-water {idx.size}  norm of row 0 {idx.matrix[0].norm():.4f}")

    q = "which region does atlas deploy to"
    print(f"\nHYBRID RETRIEVAL  query: {q!r}  now=6.0  half_life=3")
    print(f"  {'S':>6} {'sem':>6} {'rec':>6} {'imp':>6}  memory")
    for h in mem.retrieve(q, now=6.0, k=5):
        print(f"  {h.score:>6.3f} {h.semantic:>6.3f} {h.recency:>6.3f} {h.importance:>6.3f}  {h.item.text}")

    print("\nCONSOLIDATION")
    for i, (proj, err, fix) in enumerate([("atlas", "MigrationError", "migrate"), ("cobalt", "MigrationError", "migrate"),
                                         ("dune", "AuthError", "login")]):
        task = f"t{i}"
        steps = [(f"deploy {proj}", f"{err}: failed", -1.0), (f"clear_cache {proj}", "ok", 0.0),
                 (f"deploy {proj}", f"{err}: failed", -1.0), (f"{fix} {proj}", "ok", 0.0), (f"deploy {proj}", "healthy", 1.0)]
        for j, (a, o, r) in enumerate(steps):
            mem.record_step(Episode(f"deploy {proj}", a, o, r, 5.0 + i + j / 100, task, 5 + i, proj))
        item = mem.end_task(task)
        print(" ", item.text)
    for p in mem.end_session():
        print("  ->", p.text())
    print("  dune alone gave no rule: one project is a quirk, not a pattern")

    print("\nCONTEXT PACKER, B=160")
    mem.working = TaskState("deploy atlas", "run the deploy", open_questions=["which region?"])
    packed = mem.build_context("You are an ops agent.", "Deploy atlas.", now=8.0, budget=160,
                               recent_events=["[s8t1] user asked for a deploy", "[s8t1] `deploy atlas` -> AuthError"],
                               observation="\n".join(f"[{i}/7] step done in 1.{i}s" for i in range(1, 8)) + "\nAuthError: registry rejected the push token (401)")
    print(f"  tokens by section {packed.tokens_by_section}  total {packed.total}  rendered {count_tokens(packed.render())}")
    print(f"  dropped {[b.id for b in packed.dropped]}  compressed {packed.compressed}")
    print("  ---")
    print("\n".join("  " + line for line in packed.render().splitlines()))


if __name__ == "__main__":
    main()
