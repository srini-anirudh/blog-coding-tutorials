"""Put the lab folder on sys.path so scripts run from anywhere without installing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rooflab.hardware import be_polite  # noqa: E402

be_polite()
