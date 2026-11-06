"""Put the lab folder on sys.path, so scripts run from the repository root without installing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnmoves.common import polite  # noqa: E402

polite()
