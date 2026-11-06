"""Turn a model's free-text answer into a closing time, and score it."""
from __future__ import annotations

import re
from collections import Counter

QUESTION = "When does the bakery close? Answer with the closing time only, in 24-hour HH:MM format."
COT = ("When does the bakery close? First list every sign in the image and the text on it. "
       "Then reason step by step. End with 'Answer: HH:MM' in 24-hour format.")
CAPTION = "Read out every piece of text you can see in this image, exactly as written."
FIND = "Is there a small sign with a shop's opening hours in this image? Answer Yes or No."
VERIFY_GENERIC = ("Look at the text in this image. Does it give a shop's opening hours (not a bus or train time)? "
                  "Answer Yes or No.")
# A narrower relevance check. VERIFY_GENERIC gave P(Yes) 0.57 on a bus sign, above a 0.5 threshold, so it rejected
# nothing (see results/06_search.txt). This wording was picked on scenes 2000-2039 and is used on scenes 5000 and up.
VERIFY = "Does this image show a sign that says OPEN followed by opening hours? Answer Yes or No."
_TIME = re.compile(r"\b(\d{1,2})[:.](\d{2})\s*([AaPp]\.?\s?[Mm]\.?)?")


def times_in(text: str) -> list[str]:
    """Every time in the text, normalised to H:MM in 24-hour form ("5:30 pm" -> "17:30")."""
    out = []
    for h, m, ampm in _TIME.findall(text):
        h = int(h)
        if ampm and ampm[0] in "pP" and h < 12:
            h += 12
        if h < 24 and int(m) < 60:
            out.append(f"{h}:{m}")
    return out


def closing_time(text: str) -> str | None:
    """The closing time the answer states. For a range like "9:00-17:30" that is the later time."""
    ts = times_in(text)
    if not ts:
        return None
    if len(ts) >= 2 and re.search(r"\d\s*(-|–|to|until)\s*\d", text):
        return max(ts[-2:], key=lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1]))
    return ts[-1]


def majority(answers: list[str | None]) -> str | None:
    """Majority vote over parsed answers. Unparsed answers do not vote. Ties go to the earliest."""
    votes = [a for a in answers if a is not None]
    if not votes:
        return None
    counts = Counter(votes)
    best = max(counts.values())
    return next(a for a in votes if counts[a] == best)


def weighted_majority(answers: list[str | None], weights: list[float]) -> str | None:
    """Vote where each answer counts for its weight. Unparsed answers do not vote."""
    tally: dict[str, float] = {}
    for a, w in zip(answers, weights):
        if a is not None:
            tally[a] = tally.get(a, 0.0) + w
    return max(tally, key=tally.get) if tally else None
