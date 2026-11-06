"""Figure 9 and sections 10 and 17: append, replace or supersede; two clocks; deletion that chases copies.

    python labs/memory-is-more-than-context/scripts/06_bitemporal.py
"""

from __future__ import annotations

from datetime import date as D

from _common import Report

from memlab.bitemporal import AppendStore, BitemporalStore, ReplaceStore

out = Report("06_bitemporal")

# The story, in the order the system hears it.
EVENTS = [
    # (subject, relation, value, true in the world from, learned on, source)
    ("user", "lives_in", "Hyderabad", D(2023, 6, 1), D(2023, 6, 1), "user"),
    ("project-x", "deploys_via", "Jenkins", D(2025, 1, 10), D(2025, 1, 10), "user"),
    ("project-x", "deploys_via", "GitHub Actions", D(2026, 8, 14), D(2026, 8, 14), "user"),
    ("user", "diet", "vegetarian", D(2024, 2, 1), D(2024, 2, 1), "user"),
    # The user moved on 15 July and only says so on 10 September: the two clocks come apart.
    ("user", "lives_in", "Bangalore", D(2026, 7, 15), D(2026, 9, 10), "user"),
    # A model inference that turns out to be wrong.
    ("project-x", "runner_image", "v2", D(2026, 8, 20), D(2026, 8, 20), "model-inferred"),
]
TODAY = D(2026, 10, 1)

stores = {"append-only": AppendStore(), "replace": ReplaceStore(), "bitemporal": BitemporalStore()}
facts: dict[tuple, int] = {}
for s, r, v, vf, rec, src in EVENTS:
    for name, st in stores.items():
        fid = st.tell(s, r, v, valid_from=vf, recorded_at=rec, source=src)
        if name == "bitemporal":
            facts[(s, r, v)] = fid
bt: BitemporalStore = stores["bitemporal"]
bt.derive("profile: lives in Bangalore, vegetarian, prefers long-form writing",
          [facts[("user", "lives_in", "Bangalore")], facts[("user", "diet", "vegetarian")]])
# 25 Sept: a tool check shows the runner image claim was never true.
bt.invalidate(facts[("project-x", "runner_image", "v2")], at=D(2026, 9, 25))

out("Operation log of the bitemporal store:")
for line in bt.log:
    out("  " + line)

QUESTIONS = [
    ("Where does the user live now?", "current", ("user", "lives_in"), ["Bangalore"]),
    ("How does project-x deploy now?", "current", ("project-x", "deploys_via"), ["GitHub Actions"]),
    ("Where did the user live on 1 Jan 2026?", ("as_of", D(2026, 1, 1), None), ("user", "lives_in"), ["Hyderabad"]),
    ("Where did the user actually live on 1 Aug 2026?", ("as_of", D(2026, 8, 1), None), ("user", "lives_in"),
     ["Bangalore"]),
    ("On 1 Aug 2026, where did we believe the user lived?", ("as_of", D(2026, 8, 1), D(2026, 8, 1)),
     ("user", "lives_in"), ["Hyderabad"]),
    ("Which runner image does project-x use now?", "current", ("project-x", "runner_image"), []),
    ("On 21 Sep 2026, what did we believe the runner image was?", ("as_of", D(2026, 9, 21), D(2026, 9, 21)),
     ("project-x", "runner_image"), ["v2"]),
]


def answer(name, st, kind, slot):
    if kind == "current":
        if name == "bitemporal":
            return st.current(*slot, today=TODAY)
        return st.current(*slot)
    if name != "bitemporal":
        return None  # no history kept: the question cannot be asked
    _, world_t, known_at = kind
    return st.as_of(*slot, world_time=world_t, known_at=known_at or TODAY)


# Append and replace stores have no INVALIDATE; the closest they can do is nothing.
out(f"\n{'question':<58}{'gold':<18}" + "".join(f"{n:<26}" for n in stores))
score = {n: 0 for n in stores}
for q, kind, slot, gold in QUESTIONS:
    cells = []
    for name, st in stores.items():
        got = answer(name, st, kind, slot)
        ok = got is not None and sorted(got) == sorted(gold)
        score[name] += ok
        cells.append(("cannot ask" if got is None else (", ".join(got) or "(none)")) + (" ok" if ok else " X"))
    out(f"{q:<58}{', '.join(gold) or '(none)':<18}" + "".join(f"{c:<26}" for c in cells))
out("correct: " + ", ".join(f"{n} {s}/{len(QUESTIONS)}" for n, s in score.items()))

out("\nDeletion: the user asks the agent to forget their diet.")
for chase in (False, True):
    st = BitemporalStore()
    for s, r, v, vf, rec, src in EVENTS:
        st.tell(s, r, v, valid_from=vf, recorded_at=rec, source=src)
    diet = next(r.fact for r in st.rows if r.relation == "diet")
    city = next(r.fact for r in st.rows if r.value == "Bangalore")
    st.derive("profile: lives in Bangalore, vegetarian, prefers long-form writing", [city, diet])
    st.delete(diet, chase_derived=chase)
    leaked = "vegetarian" in st.visible_text(TODAY)
    out(f"  delete {'and chase derived records' if chase else 'the fact row only':<32} "
        f"'vegetarian' still visible to a future context: {leaked}")
out.save({"score": score, "questions": len(QUESTIONS)})
