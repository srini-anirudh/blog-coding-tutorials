"""Mechanical checks for arithmetic in free-text reasoning traces (used on GSM8K-style solutions).

We look for statements of the form `a op b (op c ...) = d`, normalise LaTeX operators, evaluate the left side with the
usual precedence and compare with d. Only plain arithmetic statements are checked; reasoning about what to compute is not."""

from __future__ import annotations

import re

NUM = r"\d[\d,]*\.?\d*"
EQ = re.compile(rf"({NUM}(?:\s*[+\-*/]\s*{NUM})+)\s*=\s*({NUM})(?![\d.,]*\s*[+\-*/]\s*\d)(?!\s*percent)")


def normalise(text: str) -> str:
    t = text.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/").replace("×", "*").replace("÷", "/")
    t = re.sub(r"\\text\{[^}]*\}", " ", t)
    t = re.sub(r"(\d)\s*\\?%", r"\1 percent", t)          # percent arithmetic is not checked
    t = re.sub(r"\\[()\[\]]|\\,|\\;|\\!|\\\$|\$|\{,\}", " ", t)
    return t.replace("\\", " ")


def _val(s: str) -> float:
    return float(s.replace(",", "").rstrip("."))


def equations(text: str) -> list[tuple[str, float, float]]:
    """(expression, computed value, claimed value) for each arithmetic statement found."""
    out = []
    t = normalise(text)
    for m in EQ.finditer(t):
        lhs, rhs = m.group(1), m.group(2)
        before, after = t[:m.start()].rstrip(), t[m.end():].lstrip()
        # part of a larger expression (an operator or a bracket on either side): not a complete statement
        if (before and before[-1] in "+-*/(^") or after[:1] in (")", "^"):
            continue
        expr = re.sub(r"(\d[\d,]*\.?\d*)", lambda m: str(_val(m.group())), lhs)
        try:
            v = eval(expr, {"__builtins__": {}}, {})  # only digits, dots, spaces and + - * / reach here
        except Exception:
            continue
        out.append((lhs.strip(), float(v), _val(rhs)))
    return out


def holds(eq: tuple[str, float, float], rel: float = 0.01) -> bool:
    _, v, c = eq
    return abs(v - c) <= max(0.011, rel * abs(v))      # allow rounding of the written result


def n_wrong(text: str) -> tuple[int, int]:
    """(number of arithmetic statements found, number that are false)."""
    eqs = equations(text)
    return len(eqs), sum(not holds(e) for e in eqs)


def path_key(text: str) -> tuple:
    """Distinct-path signature: the ordered list of statements with numbers rounded."""
    return tuple((e[0].replace(" ", ""), round(e[2], 3)) for e in equations(text))
