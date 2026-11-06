"""Device selection and synchronisation.

`x.cuda()` is shorthand for `x.to("cuda")`. Both return a copy on the GPU and
leave `x` where it was. We use `.to(device)` so the same code runs on any device.

GPU work is queued, not executed, when Python returns. Any timing code must
call `synchronize(device)` before reading the clock.
"""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device: cuda > mps > cpu.

    `prefer` forces a device by name (for example "cpu" in tests).
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device | str) -> None:
    """Block until all queued work on `device` has finished."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    # CPU ops run eagerly, so there is nothing to wait for.
