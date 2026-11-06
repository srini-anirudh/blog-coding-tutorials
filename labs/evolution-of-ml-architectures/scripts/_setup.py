"""Put the lab folder on sys.path, so scripts run from anywhere without installing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
