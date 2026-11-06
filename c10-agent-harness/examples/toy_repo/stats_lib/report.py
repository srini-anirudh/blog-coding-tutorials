from .core import mean, median, variance


def summarize(xs: list[float]) -> dict[str, float]:
    return {"n": len(xs), "mean": mean(xs), "median": median(xs), "variance": variance(xs)}
