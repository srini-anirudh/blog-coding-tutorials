"""Multiple choice: options, prompts, answer extractors and CircularEval.

The post's sections "The answer was right but scored wrong" and "The model prefers a position,
not an answer". Three extractors stand in for the range of parsers real toolkits use:

strict      the reply must be a bare option letter, such as "B" or "(B)."
permissive  find exactly one standalone option letter anywhere in the reply
content     permissive, then fall back to matching the text of exactly one option

None of them calls a language model. VLMEvalKit and MathVista add an LLM extractor as a last
step; we do not have one here, so an unparsed reply simply scores 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LETTERS = "ABCDEFGHIJ"
FORMAT_HINT = "Answer with the option's letter from the given choices directly."


@dataclass(frozen=True)
class MCItem:
    question: str          # the stem, without the options
    options: list[str]     # option texts, in the benchmark's order
    answer: int            # index of the correct option

    @property
    def gold_letter(self) -> str:
        return LETTERS[self.answer]


_OPT = re.compile(r"(?:^|,\s*|\s)([A-J]):\s")


_CHOICE = re.compile(r"^\(([A-J])\)\s?(.*)$")


def _parse_choices_block(question: str, answer: str) -> MCItem:
    """The MathVista layout MMStar keeps for its maths items: 'Hint: ...\\nQuestion: ...\\nChoices:\\n(A) x'."""
    all_lines = question.splitlines()
    start = next((i for i, ln in enumerate(all_lines) if ln.strip().startswith("(A)")), None)
    if start is None:
        raise ValueError("no '(A)' choice line")
    head = "\n".join(all_lines[:start]).replace("Choices:", "")
    lines = [ln.strip() for ln in all_lines[start:] if ln.strip()]
    pairs = [_CHOICE.match(ln) for ln in lines]
    if not pairs or not all(pairs):
        raise ValueError("unreadable 'Choices:' block")
    letters = [m.group(1) for m in pairs]
    if letters != list(LETTERS[:len(letters)]) or len(letters) < 2 or answer not in letters:
        raise ValueError(f"choices are not A, B, C, ... in order: {letters}")
    stem = head.strip()
    if "Question:" in stem:  # drop the MathVista format hint; our prompt adds its own
        stem = stem.split("Question:", 1)[1].strip()
    return MCItem(stem, [m.group(2).strip() for m in pairs], letters.index(answer))


def parse_mmstar(question: str, answer: str) -> MCItem:
    """Split an MMStar question into stem and options.

    Most items use '...\\nOptions: A: x, B: y, C: z, D: w'. The maths items taken from MathVista
    use 'Choices:\\n(A) x\\n(B) y'. Both become the same MCItem, and format_prompt writes every
    item in the first layout, so all items share one prompt template.
    """
    if "Options:" not in question:
        return _parse_choices_block(question, answer)
    stem, sep, opts = question.rpartition("Options:")
    if not sep:
        raise ValueError("no 'Options:' block")
    marks = list(_OPT.finditer(" " + opts.strip()))
    letters = [m.group(1) for m in marks]
    if letters != list(LETTERS[:len(letters)]) or len(letters) < 2:
        raise ValueError(f"options are not A, B, C, ... in order: {letters}")
    text = " " + opts.strip()
    options = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        options.append(text[m.end():end].strip().rstrip(","))
    if answer not in letters:
        raise ValueError(f"gold letter {answer!r} not among {letters}")
    return MCItem(stem.strip(), options, letters.index(answer))


def rotate(item: MCItem, k: int) -> MCItem:
    """Shift the options by k positions: the option at index i moves to index (i - k) mod N."""
    n = len(item.options)
    opts = [item.options[(i + k) % n] for i in range(n)]
    return MCItem(item.question, opts, (item.answer - k) % n)


def circular_variants(item: MCItem) -> list[MCItem]:
    """The N passes of MMBench's CircularEval: the same question with the options rotated."""
    return [rotate(item, k) for k in range(len(item.options))]


def format_prompt(item: MCItem, hint: bool = True) -> str:
    """The MMStar layout: stem, then 'Options: A: ..., B: ...', then an optional format hint."""
    opts = ", ".join(f"{LETTERS[i]}: {o}" for i, o in enumerate(item.options))
    text = f"{item.question}\nOptions: {opts}"
    return f"{text}\n{FORMAT_HINT}" if hint else text


# --- Extractors ------------------------------------------------------------------------------

def extract_strict(reply: str, n_options: int) -> str | None:
    m = re.fullmatch(r"\s*\(?([A-J])\)?[.:]?\s*", reply)
    return m.group(1) if m and m.group(1) in LETTERS[:n_options] else None


def extract_permissive(reply: str, n_options: int) -> str | None:
    """A letter that stands alone ('B', '(B)', 'B.', 'answer is B'), if exactly one distinct one does.

    "A" is also an English word, so a reply like "A dog on a sofa" is ambiguous. We treat a
    capital A as a letter only when it is followed by punctuation or ends the reply, the same
    kind of heuristic rule-based extractors use. The ambiguity is the point: rules guess.
    """
    valid = LETTERS[:n_options]
    found = []
    for m in re.finditer(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", reply):
        c = m.group(1)
        if c not in valid:
            continue
        after = reply[m.end():m.end() + 1]
        if c == "A" and after == " " and m.start() == 0 and not re.match(r"A\s*[:.)]", reply):
            continue  # "A dog ..." at the start of a sentence is an article
        found.append(c)
    distinct = set(found)
    return found[0] if len(distinct) == 1 else None


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def extract_content(reply: str, options: list[str]) -> str | None:
    """Permissive letter match, else the one option whose text appears in the reply."""
    letter = extract_permissive(reply, len(options))
    if letter is not None:
        return letter
    r = _norm(reply)
    if not r:
        return None
    hits = [i for i, o in enumerate(options) if _norm(o) and (_norm(o) in r or r == _norm(o))]
    return LETTERS[hits[0]] if len(hits) == 1 else None


EXTRACTORS = ("strict", "permissive", "content")


def extract(reply: str, item: MCItem, how: str) -> str | None:
    if how == "strict":
        return extract_strict(reply, len(item.options))
    if how == "permissive":
        return extract_permissive(reply, len(item.options))
    if how == "content":
        return extract_content(reply, item.options)
    raise ValueError(how)


def circular_correct(picks: list[str | None], variants: list[MCItem]) -> bool:
    """CircularEval credit: right on every rotation."""
    return all(p == v.gold_letter for p, v in zip(picks, variants, strict=True))
