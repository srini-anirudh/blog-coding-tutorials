"""Device selection shared by every script: CUDA if present, then Apple MPS, then CPU."""

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or `prefer` if it is given ("cpu", "mps", "cuda")."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """A short label for result tables, e.g. "MPS" or "CPU"."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type.upper()
