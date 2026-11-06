"""Small descriptive statistics, written the way real code drifts: mostly right."""


def mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("mean of empty list")
    return sum(xs) / len(xs)


def median(xs: list[float]) -> float:
    if not xs:
        raise ValueError("median of empty list")
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid]


def variance(xs: list[float]) -> float:
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)
